import random
import numpy as np
from genedesign.models.transcript import Transcript
from genedesign.rbs_chooser import RBSChooser
from genedesign.checkers.codon_checker import CodonChecker
from genedesign.checkers.hairpin_checker import HairpinChecker
from genedesign.checkers.internal_promoter_checker import InternalPromoterChecker
from genedesign.checkers.forbidden_sequence_checker import ForbiddenSequenceChecker
from genedesign.checkers.g_quadruplex_checker import GQuadruplexChecker # Your new checker

class TranscriptDesigner:
    """
    An optimized designer that balances high CAI with biological safety constraints
    (hairpins, G-quadruplexes, promoters) using a Hill Climbing algorithm.
    """

    def __init__(self):
        self.rbs_chooser = RBSChooser()
        self.codon_checker = CodonChecker()
        self.hairpin_checker = HairpinChecker()
        self.promoter_checker = InternalPromoterChecker()
        self.forbidden_checker = ForbiddenSequenceChecker()
        self.g4_checker = GQuadruplexChecker()

    def initiate(self) -> None:
        """
        Initializes all underlying models and checkers.
        """
        self.rbs_chooser.initiate()
        # Note: Depending on your repo, some checkers may need .initiate() calls
        # if they load external data files like codon_usage.txt.

    def run(self, peptide: str, host, iterations: int = 500) -> Transcript:
        """
        Translates a peptide into an optimized DNA sequence.
        """
        # 1. Start with the "Greedy" best-case CAI sequence
        # We use the host-specific optimal codons as a starting point.
        cds_list = [self.codon_checker.get_optimal_codon(aa, host) for aa in peptide]
        cds_list.append("TAA") # Adding the stop codon
        cds = "".join(cds_list)

        # 2. Choose the initial RBS
        # We assume the user wants to ignore an empty set of RBS options for now.
        selected_rbs = self.rbs_chooser.run(cds, set())
        
        best_full_seq = selected_rbs.utr + cds
        best_score = self._calculate_score(best_full_seq, host)

        # 3. Optimization Loop: Stochastic Hill Climbing
        # This replaces the static dictionary approach with a flexible search.
        for _ in range(iterations):
            # Pick a random amino acid position to mutate (excluding the stop codon)
            aa_idx = random.randint(0, len(peptide) - 1)
            target_aa = peptide[aa_idx]
            
            # Get synonymous alternatives for this specific amino acid
            alt_codons = self.codon_checker.get_synonymous_codons(target_aa, host)
            if len(alt_codons) <= 1:
                continue
                
            new_codon = random.choice(alt_codons)
            
            # Create a mutated candidate CDS
            start_idx = aa_idx * 3
            mutated_cds_list = list(cds_list)
            mutated_cds_list[aa_idx] = new_codon
            mutated_cds = "".join(mutated_cds_list)
            
            mutated_full_seq = selected_rbs.utr + mutated_cds
            
            # Evaluate the fitness of this new candidate
            new_score = self._calculate_score(mutated_full_seq, host)
            
            # If the score is better (or equal), accept the change
            if new_score >= best_score:
                cds_list = mutated_cds_list
                cds = mutated_cds
                best_full_seq = mutated_full_seq
                best_score = new_score

        # 4. Final return
        return Transcript(selected_rbs, peptide, cds_list)

    def _calculate_score(self, sequence: str, host) -> float:
        """
        Computes the fitness of a sequence based on CAI and biological penalties.
        """
        # Positive Metric: Codon Adaptation Index
        cai = self.codon_checker.calc_cai(sequence, host)
        score = cai * 100 
        
        # Penalty 1: Hairpins (Secondary Structure)
        # Hairpins stall ribosomes; we penalize them per occurrence.
        hairpin_count = self.hairpin_checker.run(sequence)
        score -= (hairpin_count * 15)
        
        # Penalty 2: G-Quadruplexes (Your New Integration)
        # These non-B DNA knots can stop transcription entirely.
        if self.g4_checker.run(sequence):
            score -= 500  # Extreme penalty to force synonymous codon swapping
            
        # Hard Constraints: Forbidden Sites & Internal Promoters
        # If these exist, the sequence is effectively "dead" for a MoClo assembly.
        if self.forbidden_checker.run(sequence) or self.promoter_checker.run(sequence):
            score -= 2000 
            
        return score

if __name__ == "__main__":
    # Test execution
    from genedesign.models.host import Host
    peptide = "MYPFIRTARMTV"
    
    designer = TranscriptDesigner()
    designer.initiate()
    
    # Running for E. coli with 1000 iterations for high-quality design
    result = designer.run(peptide, Host.Ecoli, iterations=1000)
    
    print(f"Designed CDS: {result.cds}")
    print(f"RBS Used: {result.rbs.utr}")
