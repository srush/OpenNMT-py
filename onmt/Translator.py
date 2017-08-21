import onmt
import onmt.Models
import onmt.modules
import onmt.IO
import torch
from torch.autograd import Variable
import dill


def make_features(batch, fields):
    # This is a bit hacky for now.
    feats = []
    for j in range(100):
        key = "src_feat_" + str(j)
        if key not in fields:
            break
        feats.append(batch.__dict__[key])
    cat = [batch.src[0]] + feats
    cat = [c.unsqueeze(2) for c in cat]
    return torch.cat(cat, 2)


class Translator(object):
    def __init__(self, opt, dummy_opt={}):
        # Add in default model arguments, possibly added since training.
        self.opt = opt
        checkpoint = torch.load(opt.model,
                                map_location=lambda storage, loc: storage,
                                pickle_module=dill)
        self.fields = checkpoint['fields']
        model_opt = checkpoint['opt']
        for arg in dummy_opt:
            if arg not in model_opt:
                model_opt.__dict__[arg] = dummy_opt[arg]

        self._type = model_opt.encoder_type
        self.copy_attn = model_opt.copy_attn

        self.model = onmt.Models.make_base_model(opt, model_opt, self.fields,
                                                 opt.cuda, checkpoint)
        self.model.eval()

        # for debugging
        self.beam_accum = None

    def initBeamAccum(self):
        self.beam_accum = {
            "predicted_ids": [],
            "beam_parent_ids": [],
            "scores": [],
            "log_probs": []}

    def buildTargetTokens(self, pred, src, attn, copy_vocab):
        vocab = self.fields["tgt"].vocab
        tgt_eos = vocab.stoi[onmt.IO.EOS_WORD]
        tokens = []
        for tok in pred:
            if tok < len(vocab):
                tokens.append(vocab.itos[tok])
            else:
                tokens.append(copy_vocab.itos[tok - len(vocab)])

        tokens = tokens[:-1]  # EOS
        if self.opt.replace_unk:
            for i in range(len(tokens)):
                if tokens[i] == onmt.IO.UNK:
                    _, maxIndex = attn[i].max(0)
                    tokens[i] = src[maxIndex[0]]
        return tokens

    def _runTarget(self, batch, data):
        if "tgt" not in batch.__dict__:
            return None

        _, src_lengths = batch.src
        src = make_features(batch, self.fields)

        #  (1) run the encoder on the src
        encStates, context = self.model.encoder(src, src_lengths)
        decStates = self.model.init_decoder_state(context, encStates)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        goldScores = 0
        decOut, decStates, attn = self.model.decoder(
            batch.tgt[:-1], batch.src, context, decStates)

        aeq(decOut.size(), batch.tgt[1:].data.size())
        for dec, tgt in zip(decOut, batch.tgt[1:].data):
            # Log prob of each word.
            out = self.model.generator.forward(dec)
            tgt = tgt.unsqueeze(1)
            scores = out.data.gather(1, tgt)
            scores.masked_fill_(tgt.eq(tgt_pad), 0)
            goldScores += scores[0]
        return goldScores


    def translateBatch(self, batch, dataset):
        beamSize = self.opt.beam_size

        #  (1) run the encoder on the src
        _, src_lengths = batch.src
        src = make_features(batch, self.fields)
        encStates, context = self.model.encoder(src, src_lengths)
        decStates = self.model.init_decoder_state(context, encStates)

        #  (1b) initialize for the decoder.
        def var(a): return Variable(a, volatile=True)
        def rvar(a): return var(a.repeat(1, beamSize, 1))

        # Repeat everything beam_times
        context = rvar(context.data)
        src = rvar(src.data)
        src_map = rvar(batch.src_map.data)
        decStates.repeatBeam_(beamSize)
        beam = onmt.Beam(beamSize, n_best=self.opt.n_best, cuda=self.opt.cuda,
                         vocab=self.fields["tgt"].vocab)


        #  (2) run the decoder to generate sentences, using beam search
        for i in range(self.opt.max_sent_length):
            # Construct batch x beam_size next words.
            inp = var(beam.getCurrentState().unsqueeze(0))
            inp = inp.masked_fill(inp.gt(len(self.fields["tgt"].vocab) - 1), 0) # 0 is unk
            # 1 x beam_size

            # Run one step.
            decOut, decStates, attn = \
                self.model.decoder(inp, src, context, decStates)
            decOut = decOut.squeeze(0)
            # decOut: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            if not self.copy_attn:
                out = self.model.generator.forward(decOut).data
                # beam x tgt_vocab
            else:
                attn_copy = attn["copy"].squeeze(0).contiguous()
                
                out = self.model.generator.forward(
                    decOut, attn_copy, src_map)
                # beam x (tgt_vocab + extra_vocab)
                # out = out.data.unsqueeze(1)
                out = dataset.collapseCopyScores(
                    out.data.unsqueeze(1), batch, self.fields["tgt"].vocab)
                # beam x tgt_vocab
                out = out.squeeze(1).log()

            # (c) Advance each beam.
            is_done = beam.advance(out, attn["copy"].data.squeeze(0))
            decStates.beamUpdate_(beam.getCurrentOrigin())
            if is_done:
                break

        #  (3) package everything up
        n_best = self.opt.n_best
        scores, ks = beam.sortFinished()
        hyps, attn = [], []
        for i, (time, k, cb) in enumerate(ks):# [:n_best]:
            # print(i, scores[i], cb)
            hyp, att = beam.getHyp(time, k)
            hyps.append(hyp)
            attn.append(att)

        return [hyps], [scores], [attn], [0]

    def translate(self, batch, data):
        #  (1) convert words to indexes
        batchSize = batch.batch_size

        #  (2) translate
        pred, predScore, attn, goldScore = self.translateBatch(batch, data)
        pred, predScore, attn, goldScore, i = list(zip(
            *sorted(zip(pred, predScore, attn, goldScore,
                        batch.indices.data),
                    key=lambda x: x[-1])))
        inds, perm = torch.sort(batch.indices.data)

        #  (3) convert indexes to words
        predBatch = []
        src = batch.src[0].data.index_select(1, perm)
        for b in range(batchSize):
            src_vocab = data.src_vocabs[inds[b]]
            predBatch.append(
                [self.buildTargetTokens(pred[b][n], src[:, b],
                                        attn[b][n], src_vocab)
                 for n in range(self.opt.n_best)]
            )

        return predBatch, predScore, goldScore, attn, src
