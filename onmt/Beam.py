from __future__ import division
import torch
import onmt

"""
 Class for managing the internals of the beam search process.

 Takes care of beams, back pointers, and scores.
"""


class Beam(object):
    def __init__(self, size, n_best=1, cuda=False, vocab=None):

        self.size = size
        self.done = False

        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.allScores = []

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(vocab.stoi[onmt.IO.PAD_WORD])]
        self.nextYs[0][0] = vocab.stoi[onmt.IO.BOS_WORD]
        self.vocab = vocab

        # The attentions (matrix) for each time.
        self.attn = []
        
        # Sum of attentions at beam. 
        self.coverage = None
        self.sents = None 

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best
        
    def getCurrentState(self):
        "Get the outputs for the current timestep."
        return self.nextYs[-1]

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def _global_score(self, end=False, beta = 0.15):
        # pen = self.tt.FloatTensor(self.nextYs[-1].size(0)).fill_(0)
        pen = beta * torch.min(1.0 - self.coverage,
                               self.coverage.clone().fill_(0.0)).sum(1).squeeze(1)
        for i in range(self.nextYs[-1].size(0)):
            if not end and self.nextYs[-1][i]  == self.vocab.stoi[onmt.IO.EOS_WORD]:
                pen[i] = -1e20
            
            if self.sents[i] < 3:
                pen[i] -= 1.0
        return pen
    
    def advance(self, wordLk, attnOut, alpha = 0.9):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        pen = None
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)
            pen = self._global_score()
            beamLk.add_(pen.unsqueeze(1).expand_as(beamLk))
        else:
            beamLk = wordLk[0]
            
        flatBeamLk = beamLk.view(-1)


        
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)
        self.allScores.append(self.scores)
        self.scores = bestScores 

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId / numWords
        self.prevKs.append(prevK)
        if pen is not None:
            self.scores = self.scores - pen[prevK]
        self.nextYs.append(bestScoresId - prevK * numWords)
        self.attn.append(attnOut.index_select(0, prevK))

        if self.coverage is None:
            self.coverage = self.attn[-1]
        else:
            self.coverage = self.coverage.index_select(0, prevK).add(self.attn[-1])

        if self.sents is None:
            self.sents = self.nextYs[-1].eq(self.vocab.stoi["</t>"])
        else:
            self.sents = self.sents.index_select(0, prevK).add(self.nextYs[-1].eq(self.vocab.stoi["</t>"]))
            
        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self.vocab.stoi[onmt.IO.EOS_WORD]:
                k = i
                # Ranking score from Wu et al.
                pen = self._global_score(end=True)
                s = self.scores[i] / ((5 + len(self.nextYs)) ** alpha / (5 + 1) ** alpha)                
                coverage_bonus = pen[i] if pen is not None else 0
                s += coverage_bonus
                self.finished.append((s, len(self.nextYs) - 1, i, coverage_bonus))

        # End condition is when top-of-beam is EOS.
        # if self.nextYs[-1][0] == self.vocab.stoi[onmt.IO.EOS_WORD] and len(self.finished) > self.n_best:
        #     self.done = True
        #     self.allScores.append(self.scores)

            
        return self.done

    def sortFinished(self):
        # print(len(self.finished))
        self.finished.sort(key=lambda a: -a[0])
        scores = [s for s, _, _, cb in self.finished]
        ks = [(t, k, cb) for _, t, k, cb in self.finished]
        return scores, ks
        
    def getHyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp, attn = [], []
        for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
            hyp.append(self.nextYs[j+1][k])
            attn.append(self.attn[j][k])
            k = self.prevKs[j][k]

        return hyp[::-1], torch.stack(attn[::-1])
