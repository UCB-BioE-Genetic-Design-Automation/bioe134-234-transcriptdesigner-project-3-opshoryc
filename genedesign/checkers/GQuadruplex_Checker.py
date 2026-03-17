import re

class GQuadruplexChecker:
    """
    Identifies potential G-Quadruplex (G4) structures in a DNA sequence.
    G4 structures can stall RNA polymerase and ribosomes.
    """
    
    def __init__(self):
        # Regex for 4 runs of at least 3 Guanines with 1-7 bp loops in between
        self.g4_pattern = re.compile(r'G{3,}.{1,7}G{3,}.{1,7}G{3,}.{1,7}G{3,}', re.IGNORECASE)

    def run(self, sequence: str) -> bool:
        """
        Returns True if a G-quadruplex motif is detected, False otherwise.
        """
        if not sequence:
            return False
            
        # Search for the G4 pattern
        match = self.g4_pattern.search(sequence)
        
        return bool(match)
