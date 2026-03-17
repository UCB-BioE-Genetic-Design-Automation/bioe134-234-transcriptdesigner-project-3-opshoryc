import random
from genedesign.models.transcript import Transcript
from genedesign.checkers.codon_checker import CodonChecker
from genedesign.checkers.hairpin_checker import HairpinChecker
from genedesign.checkers.internal_promoter_checker import InternalPromoterChecker
from genedesign.checkers.forbidden_sequence_checker import ForbiddenSequenceChecker
from genedesign.rbs_chooser import RBSChooser
from genedesign.seq_utils.reverse_complement import reverse_complement

class ImprovedTranscriptDesigner:
    def __init__(self):
        self.codon_checker = CodonChecker()
        self.hairpin_checker = HairpinChecker()
        self.promoter_checker = InternalPromoterChecker()
        self.forbidden_checker = ForbiddenSequenceChecker()
        self.rbs_chooser = RBSChooser()

    def run(self, protein_seq, host, iterations=500):
        """
        Designs a transcript with high CAI, low hairpins, and no internal 
        promoters or forbidden sequences using a Hill Climbing optimization.
        """
        # 1. Initial State: Start with a high-CAI sequence
        # (Assuming codon_checker provides a method to get the most frequent codon)
        cds = "".join([self.codon_checker.get_optimal_codon(aa, host) for aa in protein_seq])
        
        # 2. Select the optimal RBS
        rbs_option = self.rbs_chooser.run(cds, host)
        best_full_seq = rbs_option.utr + cds
        best_score = self._calculate_score(best_full_seq, host)

        # 3. Optimization Loop (Hill Climbing)
        for _ in range(iterations):
            # Randomly pick a position and swap for a synonymous codon
            aa_idx = random.randint(0, len(protein_seq) - 1)
            target_aa = protein_seq[aa_idx]
            
            # Get synonymous alternatives
            alt_codons = self.codon_checker.get_synonymous_codons(target_aa, host)
            if len(alt_codons) <= 1:
                continue
                
            new_codon = random.choice(alt_codons)
            
            # Create a mutated CDS string
            start_idx = aa_idx * 3
            mutated_cds = cds[:start_idx] + new_codon + cds[start_idx+3:]
            mutated_full_seq = rbs_option.utr + mutated_cds
            
            # Calculate new score
            new_score = self._calculate_score(mutated_full_seq, host)
            
            # If the change improves the score (or at least doesn't hurt it), keep it
            if new_score >= best_score:
                cds = mutated_cds
                best_full_seq = mutated_full_seq
                best_score = new_score

        # 4. Return the final Transcript object
        return Transcript(rbs=rbs_option, cds=cds)

    def _calculate_score(self, sequence, host):
        """
        Calculates a fitness score for a sequence.
        Weights CAI positively and penalizes hairpins/promoters/forbidden sites.
        """
        # Positive Weight: CAI (0.0 to 1.0)
        cai = self.codon_checker.calc_cai(sequence, host)
        score = cai * 100 
        
        # Penalty: Hairpins (Count)
        hairpin_count = self.hairpin_checker.run(sequence)
        score -= (hairpin_count * 10) # Heavy penalty per hairpin
        
        # Hard Penalty: Forbidden Sequences (BsaI, etc.)
        if self.forbidden_checker.run(sequence):
            score -= 1000 # "Kill" this design
            
        # Hard Penalty: Internal Promoters
        if self.promoter_checker.run(sequence):
            score -= 1000 # "Kill" this design
            
        return score
