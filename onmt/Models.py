from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import onmt
from onmt.Utils import aeq


def rnn_factory(rnn_type, **kwargs):
    # Use pytorch version when available.
    no_pack_padded_seq = False
    if rnn_type == "SRU":
        # SRU doesn't support PackedSequence.
        no_pack_padded_seq = True
        self.rnn = onmt.modules.SRU(**kwargs)
    else:
        rnn = getattr(nn, rnn_type)(**kwargs)
    return rnn, no_pack_padded_seq


class EncoderBase(nn.Module):
    """
    EncoderBase class for sharing code among various encoder.
    """
    def _check_args(self, src, lengths=None, hidden=None):
        s_len, n_batch, n_feats = src.size()
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)

    def forward(self, src, lengths=None, hidden=None):
        """
        Args:
            src (LongTensor): Src tokens [src_len x batch x nfeat].
            lengths (LongTensor): Src length [batch]
            hidden (rnn-specific): Initial hidden state.
        Returns:
            final_state: Final encoder state. Varies by type,
                  e.g. LSTM [pair of layers x batch x rnn_size]
            memory_bank (FloatTensor): Memory bank [src_len x batch x rnn_size]
        """
        raise NotImplementedError


class MeanEncoder(EncoderBase):
    """
    A trivial encoder without RNN, just takes mean as final state.

    Args:
       num_layers (int) : Number of (fake) layers.
       embeddings (Embeddings): Embedding object.
    """
    def __init__(self, num_layers, embeddings):
        super(MeanEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings

    def forward(self, src, lengths=None, hidden=None):
        """ See EncoderBase.forward() for description of args and returns. """
        self._check_args(src, lengths, hidden)

        emb = self.embeddings(src)
        s_len, batch, emb_dim = emb.size()
        mean = emb.mean(0).expand(self.num_layers, batch, emb_dim)
        encoder_final = (mean, mean)
        memory_bank = emb
        return encoder_final, memory_bank


class RNNEncoder(EncoderBase):
    """
    The standard RNN encoder.

    Args:
       rnn_type (str): RNN type.
       bidirectional (bool): Use bidirectional encoding.
       num_layers (int): Number of layers.
       hidden_size (int): Number hidden dimensions.
       dropout (float): Dropout rate.
       embeddings (Embeddings): Src embeddings to use.
    """
    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout, embeddings):
        super(RNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings
        self.rnn, self.no_pack_padded_seq = rnn_factory(rnn_type,
                    input_size=embeddings.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)

    def forward(self, src, lengths=None, hidden=None):
        """ See EncoderBase.forward() for description of args and returns."""
        self._check_args(src, lengths, hidden)

        # [s_len, batch, emb_dim]
        emb = self.embeddings(src)

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        memory_bank, encoder_final = self.rnn(packed_emb, hidden)
        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]

        return encoder_final, memory_bank


class RNNDecoderBase(nn.Module):
    """
    RNN decoder base class.

    Args:
       rnn_type (str): The type of RNN layer to use.
       bidirectional_encoder (bool): Use with bidirectional encoding.
       num_layers (int): Number of layers.
       hidden_size (int): Number hidden dimensions.
       attn_type (str): The attn type to use.
       coverage_attn (bool): Use coverage attention
       context_gate ():
       copy_attn (bool): Use copy attention.
       dropout (float): Dropout rate.
       embeddings (Embeddings): Tgt embeddings to use.
    """
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type, coverage_attn, context_gate,
                 copy_attn, dropout, embeddings):
        super(RNNDecoderBase, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type, self._input_size, hidden_size,
                                   num_layers, dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = onmt.modules.ContextGateFactory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        self.attn = onmt.modules.GlobalAttention(
            hidden_size, coverage=coverage_attn,
            attn_type=attn_type
        )

        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type
            )
            self._copy = True

    def forward(self, tgt, memory_bank, decoder_state):
        """
        Forward through the decoder.
        Args:
            tgt (LongTensor): a sequence of input tokens tensors
                              [len x batch x nfeats].
            memory_bank (FloatTensor): output from the encoder
                              [src_len x batch x hidden_size].
            decoder_state (RNNDecoderState): state initializing the decoder.
        Returns:
            decoder_outputs (FloatTensor): sequence from the decoder
                                   [len x batch x hidden_size].
            decoder_state (RNNDecoderState): Final hidden state from the decoder.
            attns (dict of (str, FloatTensor)): a dictionary of different
                                type of attention Tensor from the decoder
                                [src_len x batch].
        """
        # Check
        assert isinstance(decoder_state, RNNDecoderState)
        tgt_len, tgt_batch, _ = tgt.size()
        _, memory_batch, _ = memory_bank.size()
        aeq(tgt_batch, memory_batch)
        # END Check

        # Run the forward pass of the RNN.
        decoder_final, decoder_outputs, attns = \
            self._run_forward_pass(tgt, memory_bank, decoder_state)

        # Update the state with the result.
        final_output = decoder_outputs[-1]

        coverage = None
        if "coverge" in attns:
            coverage = attns["coverage"][-1].unsqueeze(0)

        decoder_state.update_state(decoder_final, final_output.unsqueeze(0),
                                   coverage)

        # Concatenates sequence of tensors along a new dimension.
        decoder_outputs = torch.stack(decoder_outputs)
        for k in attns:
            attns[k] = torch.stack(attns[k])
        return decoder_outputs, decoder_state, attns


    def init_decoder_state(self, src, memory_bank, encoder_final):
        """
        Initialize the decoder state based on the encoder final state.
        Args:
           src: (Not Used)
           memory_bank (FloatTensor): Memory bank [src_len x batch x rnn_size]
           encoder_final (rnn-specific): Final encoder state. Varies by type,
                  e.g. LSTM [pair of layers x batch x rnn_size]
        """
        def _fix_enc_hidden(h):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            return h

        if isinstance(encoder_final, tuple):
            return RNNDecoderState(memory_bank, self.hidden_size,
                                   tuple([_fix_enc_hidden(enc)
                                         for enc in encoder_final]))
        else:
            return RNNDecoderState(memory_bank, self.hidden_size,
                                   _fix_enc_hidden(encoder_final))


class StdRNNDecoder(RNNDecoderBase):
    """
    Stardard RNN decoder, with Attention.
    Currently no 'coverage_attn' and 'copy_attn' support.
    """
    def _run_forward_pass(self, tgt, memory_bank, state):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            tgt (LongTensor): a sequence of tgt tokens
                               [len x batch x nfeats]
            memory_bank (FloatTensor): memory bank for translation
                              [src_len x batch x hidden_size].
            state (RNNDecoderState): hidden state to begin decoder.
        Returns:
            decoder_final (RNNOutput): final hidden vector from the decoder.
            decoder_outputs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
        """
        assert not self._copy  # TODO, no support yet.
        assert not self._coverage  # TODO, no support yet.

        # Initialize local and return variables.
        attns = {}
        emb = self.embeddings(tgt)

        # Run the forward pass of the RNN.
        rnn_output, decoder_final = self.rnn(emb, state.hidden)

        # Result Check
        tgt_len, tgt_batch, _ = tgt.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(tgt_len, output_len)
        aeq(tgt_batch, output_batch)
        # END Result Check

        # Calculate the attention and the combine with rnn_output.
        query = rnn_output.transpose(0, 1).contiguous()
        decoder_outputs, p_attn = self.attn(query, memory_bank.transpose(0, 1))
        attns["std"] = p_attn

        # Calculate the context gate.
        if self.context_gate is not None:
            decoder_outputs = self.context_gate(
                emb.view(-1, emb.size(2)),
                rnn_output.view(-1, rnn_output.size(2)),
                attn_outputs.view(-1, attn_outputs.size(2))
            )
            decoder_outputs = decoder_outputs.view(tgt_len, tgt_batch, self.hidden_size)
        decoder_outputs = self.dropout(decoder_outputs)

        return decoder_final, decoder_outputs, attns

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        rnn, _ = rnn_factory(rnn_type,
                             input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             dropout=dropout)
        return rnn

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embeddings.embedding_size


class InputFeedRNNDecoder(RNNDecoderBase):
    """
    Stardard RNN decoder, with Input Feed and Attention.
    """
    def _run_forward_pass(self, tgt, memory_bank, state):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # CHECKS
        input_feed = state.input_feed.squeeze(0)
        input_feed_batch, _ = input_feed.size()
        tgt_len, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END

        # Initialize local and return variables.
        decoder_outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim
        hidden = state.hidden
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None

        # Input feed concatenates hidden state with
        # tgt at every time step.
        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)

            # Input Feed.
            decoder_input = torch.cat([emb_t, input_feed], 1)

            # RNN
            rnn_output, rnn_final = self.rnn(decoder_input, hidden)

            # Attention and context combination.
            decoder_output, p_attn = self.attn(rnn_output,
                                               memory_bank.transpose(0, 1))

            # Context gate.
            if self.context_gate is not None:
                decoder_output = self.context_gate(
                    decoder_input, rnn_output, decoder_output
                )

            decoder_output = self.dropout(decoder_output)
            decoder_outputs += [decoder_output]
            attns["std"] += [attn]

            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + attn \
                    if coverage is not None else attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy:
                _, copy_attn = self.copy_attn(decoder_output,
                                              memory_bank.transpose(0, 1))
                attns["copy"] += [copy_attn]

        # Return result.
        return rnn_final, decoder_outputs, attns

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
                "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.modules.StackedLSTM
        else:
            stacked_cell = onmt.modules.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size


class NMTModel(nn.Module):
    """
    The encoder + decoder Neural Machine Translation Model.
    """
    def __init__(self, encoder, decoder, multigpu=False):
        """
        Args:
            encoder(*Encoder): the encoder.
            decoder(*Decoder): the decoder.
            multigpu(bool): run parallel on multi-GPU?
        """
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, dec_state=None):
        """
        Args:
            src (FloatTensor): a sequence of source tensors with
                    optional feature tensors of size (len x batch).
            tgt (FloatTensor): a sequence of target tensors with
                    optional feature tensors of size (len x batch).
            lengths ([int]): an array of the src length.
            dec_state (*State): A decoder state object
        Returns:
            decoder_outputs (FloatTensor):  Decoder outputs
                    [len x batch x hidden_size]:
            attns (dict): Dictionary of (src_len x batch)
            dec_final (FloatTensor): RNN specific decoder final state.
        """
        src = src
        tgt = tgt[:-1]  # exclude last target from inputs
        encoder_final, memory_bank = self.encoder(src, lengths)
        enc_state = self.decoder.init_decoder_state(src, memory_bank, encoder_final)
        decoder_outputs, dec_final, attns = self.decoder(tgt, memory_bank,
                                             enc_state if dec_state is None
                                             else dec_state)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_final = None
            attns = None
        return decoder_outputs, attns, dec_final


class DecoderState(object):
    """
    DecoderState is a base class for models, used during translation
    for storing translation states.
    """
    def detach(self):
        """
        Detaches all Variables from the graph
        that created it, making it a leaf.
        """
        for h in self._all:
            if h is not None:
                h.detach_()

    def beam_update(self, idx, positions, beam_size):
        """ Update when beam advances. """
        for e in self._all:
            a, br, d = e.size()
            sentStates = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
            sentStates.data.copy_(
                sentStates.data.index_select(1, positions))


class RNNDecoderState(DecoderState):
    """
    Args:
       memory_bank (FloatTensor): output from the encoder of size
               len x batch x rnn_size.
       hidden_size (int): the size of hidden layer of the decoder.
       rnnstate: final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
    """
    def __init__(self, memory_bank, hidden_size, rnnstate):
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.coverage = None

        # Init the input feed.
        batch_size = memory_bank.size(1)
        h_size = (batch_size, hidden_size)
        self.input_feed = Variable(memory_bank.data.new(*h_size).zero_(),
                                   requires_grad=False).unsqueeze(0)

    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, rnnstate, input_feed, coverage):
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.input_feed = input_feed
        self.coverage = coverage

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
                for e in self._all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]
