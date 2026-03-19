import random
import re
import math
import time
from collections import Counter


from genedesign.checkers.codon_checker import CodonChecker
from genedesign.checkers.forbidden_sequence_checker import ForbiddenSequenceChecker
from genedesign.checkers.internal_promoter_checker import PromoterChecker
from genedesign.checkers.hairpin_checker import hairpin_checker
from genedesign.seq_utils.hairpin_counter import hairpin_counter
from genedesign.models.transcript import Transcript
from genedesign.rbs_chooser import RBSChooser
from genedesign.checkers.GQuadruplex_Checker import GQuadruplexChecker




class TranscriptDesigner:
   """
   Designs an optimized transcript (RBS + CDS) for a given protein sequence.


   Uses iterative stochastic optimization to satisfy multiple sequence constraints:
   codon usage (CAI), forbidden sequences, internal promoters, hairpins, and G-quadruplexes.
   """


   def __init__(self):
       self.aminoAcidToCodon: dict = {}
       self.rbsChooser = RBSChooser()
       self.codon_checker = CodonChecker()
       self.forbidden_checker = ForbiddenSequenceChecker()
       self.promoter_checker = PromoterChecker()
       self.gquad_checker = GQuadruplexChecker()
       self.relative_adaptiveness: dict = {}
       self.good_codons: dict = {}


   def initiate(self):
       """Initialize all checkers, the RBS chooser, and precompute codon tables."""
       self.codon_checker = CodonChecker()
       self.forbidden_checker = ForbiddenSequenceChecker()
       self.promoter_checker = PromoterChecker()
       self.gquad_checker = GQuadruplexChecker()
       self.rbsChooser = RBSChooser()
       self.rbsChooser.initiate()


       self.aminoAcidToCodon = {
           'A': ['GCG', 'GCC', 'GCA', 'GCT'],
           'C': ['TGC', 'TGT'],
           'D': ['GAT', 'GAC'],
           'E': ['GAA', 'GAG'],
           'F': ['TTT', 'TTC'],
           'G': ['GGC', 'GGT', 'GGA', 'GGG'],
           'H': ['CAT', 'CAC'],
           'I': ['ATT', 'ATC', 'ATA'],
           'K': ['AAA', 'AAG'],
           'L': ['CTG', 'TTA', 'TTG', 'CTT', 'CTC', 'CTA'],
           'M': ['ATG'],
           'N': ['AAC', 'AAT'],
           'P': ['CCG', 'CCA', 'CCT', 'CCC'],
           'Q': ['CAA', 'CAG'],
           'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
           'S': ['AGC', 'TCT', 'TCC', 'TCA', 'TCG', 'AGT'],
           'T': ['ACC', 'ACG', 'ACT', 'ACA'],
           'V': ['GTG', 'GTT', 'GTC', 'GTA'],
           'W': ['TGG'],
           'Y': ['TAT', 'TAC'],
           '*': ['TAA', 'TGA', 'TAG']
       }


       self.codon_checker.initiate()
       self.forbidden_checker.initiate()
       self.promoter_checker.initiate()


       self._precompute_relative_adaptiveness()
       self._precompute_good_codons()


   def _precompute_relative_adaptiveness(self):
       """Compute relative adaptiveness (w_i) for each codon."""
       self.relative_adaptiveness = {}
       for aa, codons in self.aminoAcidToCodon.items():
           if aa == '*':
               continue
           freqs = {c: self.codon_checker.codon_frequencies.get(c, 0.001) for c in codons}
           max_freq = max(freqs.values())
           if max_freq == 0:
               max_freq = 0.001
           for c in codons:
               self.relative_adaptiveness[c] = freqs[c] / max_freq


   def _precompute_good_codons(self):
       """
       Precompute a ranked list of acceptable codons per amino acid.
       Filters out rare codons and very low-frequency codons, then sorts by frequency.
       """
       self.good_codons = {}
       for aa, codons in self.aminoAcidToCodon.items():
           if aa == '*':
               continue
           safe = [c for c in codons
                   if c not in self.codon_checker.rare_codons
                   and self.codon_checker.codon_frequencies.get(c, 0.01) >= 0.15]
           if not safe:
               safe = [c for c in codons if c not in self.codon_checker.rare_codons]
           if not safe:
               safe = codons[:]
           safe.sort(key=lambda c: self.codon_checker.codon_frequencies.get(c, 0.01), reverse=True)
           self.good_codons[aa] = safe


   def compute_cai(self, codons: list) -> float:
       """Compute CAI as geometric mean of raw codon frequencies using log-space."""
       if not codons:
           return 0.0
       log_sum = 0.0
       for c in codons:
           freq = self.codon_checker.codon_frequencies.get(c, 0.01)
           if freq <= 0:
               freq = 0.01
           log_sum += math.log(freq)
       return math.exp(log_sum / len(codons))


   def initial_translate(self, protein_sequence: str, rbs_utr: str = "") -> list:
       """
       Generate an initial codon sequence prioritizing high-frequency codons
       while penalizing repeats and local hairpins, and maximizing codon diversity.
       """
       cds: list = []
       used_codons: set = set()
       check_hairpins = len(protein_sequence) < 200


       for aa in protein_sequence:
           pool = self.good_codons.get(aa.upper(), ['NNN'])
           best_codon = pool[0]
           best_score = -float('inf')


           for candidate in pool:
               repeat_penalty = 0
               if cds and cds[-1] == candidate:
                   repeat_penalty = 2
               elif len(cds) >= 2 and cds[-2] == candidate:
                   repeat_penalty = 1


               freq = self.codon_checker.codon_frequencies.get(candidate, 0.01)
               score = freq - repeat_penalty * 0.1


               if candidate not in used_codons:
                   score += 0.05


               if check_hairpins:
                   test = cds + [candidate]
                   full_so_far = rbs_utr + ''.join(test)
                   start = max(0, len(full_so_far) - 50)
                   count, _ = hairpin_counter(full_so_far[start:], 3, 4, 9)
                   score -= count * 0.03


               if score > best_score:
                   best_score = score
                   best_codon = candidate


           cds.append(best_codon)
           used_codons.add(best_codon)
       return cds


   def smart_restart(self, peptide: str) -> list:
       """Generate a randomized high-quality codon sequence for escaping local minima."""
       codons: list = []
       used_codons: set = set()
       for i, aa in enumerate(peptide):
           aa_upper = aa.upper()
           full = [c for c in self.aminoAcidToCodon.get(aa_upper, [])
                   if c not in self.codon_checker.rare_codons]
           if not full:
               full = self.aminoAcidToCodon.get(aa_upper, ['NNN'])
           options = full.copy()
           random.shuffle(options)


           unused = [c for c in options if c not in used_codons]
           if unused:
               options = unused


           if i > 0:
               recent = set(codons[max(0, i - 10):])
               preferred = [c for c in options if self._rev_comp(c) not in recent]
               if preferred:
                   options = preferred
           codons.append(options[0])
           used_codons.add(options[0])
       return codons


   @staticmethod
   def _rev_comp(seq: str) -> str:
       """Return the reverse complement of a DNA sequence."""
       comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
       return ''.join([comp.get(b, b) for b in seq.upper()[::-1]])


   def run(self, peptide: str, ignores: set) -> Transcript:
       """
       Design an optimized transcript for the given protein sequence.


       Parameters
       ----------
       peptide : str
           Amino acid sequence (without stop codon).
       ignores : set
           Set of RBS options to ignore.


       Returns
       -------
       Transcript
           A Transcript object containing the selected RBS and optimized codons.
       """
       peptide_len = len(peptide)
       clean_regex = re.compile(r'[^ATCGatcg]')


       start_time = time.time()
       TIME_LIMIT = 90


       # Phase 1: Generate initial CDS and select RBS
       initial_cds_codons = self.initial_translate(peptide)
       initial_cds_codons.append("TAA")
       initial_cds = ''.join(initial_cds_codons)


       selected_rbs = self.rbsChooser.run(initial_cds, ignores)


       if selected_rbs is None:
           raise ValueError("RBS selection failed — rbsChooser returned None")


       rbs_utr = selected_rbs.utr.upper()


       # Phase 2: Re-translate with RBS junction awareness
       best_codons = self.initial_translate(peptide, rbs_utr)
       best_codons.append("TAA")


       # Pre-compute safe mutation options per position (filtered pool)
       safe_mutation_map: dict = {}
       full_mutation_map: dict = {}
       for i, aa in enumerate(peptide):
           safe_mutation_map[i] = self.good_codons.get(aa.upper(), [best_codons[i]])
           aa_upper = aa.upper()
           full = [c for c in self.aminoAcidToCodon.get(aa_upper, [])
                   if c not in self.codon_checker.rare_codons]
           if not full:
               full = self.aminoAcidToCodon.get(aa_upper, [best_codons[i]])
           full_mutation_map[i] = full


       # --- Debug helper ---
       def make_transcript(codons, label=""):
           cds_string = ''.join(codons)
           full_transcript = rbs_utr + cds_string
           f_check = self.forbidden_checker.run(full_transcript)
           f_cds = self.forbidden_checker.run(cds_string)
           p_f_full = f_check[0] if isinstance(f_check, tuple) else f_check
           p_f_cds = f_cds[0] if isinstance(f_cds, tuple) else f_cds
           print(f"  [{label}] {peptide[:10]}... forbidden_full={p_f_full} forbidden_cds={p_f_cds}")
           if not p_f_full:
               print(f"    forbidden detail full: {f_check}")
           if not p_f_cds:
               print(f"    forbidden detail cds: {f_cds}")
           return Transcript(selected_rbs, peptide, codons)


       # --- Evaluation function ---
       def evaluate(test_codons: list, best_total_errors: int | None = None):
           cds_string = ''.join(test_codons)
           full_transcript = rbs_utr + cds_string


           c_res = self.codon_checker.run(test_codons)
           p_c = c_res[0] if isinstance(c_res, tuple) else c_res


           if isinstance(c_res, tuple) and len(c_res) >= 4:
               cai = c_res[3]
           else:
               cai = 0.0


           # Workaround for CAI underflow on long sequences
           if not p_c and isinstance(c_res, tuple) and len(c_res) >= 4:
               diversity = c_res[1]
               rare_count = c_res[2]
               reported_cai = c_res[3]
               coding = [c for c in test_codons if c not in ('TAA', 'TGA', 'TAG')]
               real_cai = self.compute_cai(coding)
               cai = real_cai
               if diversity >= 0.5 and rare_count <= 3 and reported_cai == 0.0 and real_cai >= 0.1:
                   p_c = True


           f_res = self.forbidden_checker.run(full_transcript)
           p_res = self.promoter_checker.run(full_transcript)


           gq_raw = self.gquad_checker.run(full_transcript)
           p_gq = not gq_raw
           if not p_gq:
               match = self.gquad_checker.g4_pattern.search(full_transcript)
               gq_detail = match.group() if match else ""
               gq_res = (False, gq_detail)
           else:
               gq_res = (True, "")


           p_f = f_res[0] if isinstance(f_res, tuple) else f_res
           p_p = p_res[0] if isinstance(p_res, tuple) else p_res


           errors = 0
           if not p_c:
               errors += 2
           if not p_f:
               errors += 1
           if not p_p:
               errors += 1
           if not p_gq:
               errors += 1


           if best_total_errors is not None and errors > best_total_errors:
               return errors + 999, cai, False, None


           h_res = hairpin_checker(full_transcript)
           p_h = h_res[0] if isinstance(h_res, tuple) else h_res
           if not p_h:
               chunk_size, overlap = 50, 25
               failing_chunks = 0
               for ci in range(0, len(full_transcript) - chunk_size + 1, overlap):
                   cnt, _ = hairpin_counter(full_transcript[ci:ci + chunk_size], 3, 4, 9)
                   if cnt > 1:
                       failing_chunks += 1
               errors += max(1, failing_chunks)


           passed = (errors == 0)
           return errors, cai, passed, (c_res, f_res, p_res, gq_res, h_res)


       # --- Identify problematic codon positions ---
       def get_bad_indices(test_codons: list, test_diags: tuple) -> list:
           c_res, f_res, p_res, gq_res, h_res = test_diags
           cds_string = ''.join(test_codons)
           full_transcript = rbs_utr + cds_string


           p_c = c_res[0] if isinstance(c_res, tuple) else c_res
           p_f = f_res[0] if isinstance(f_res, tuple) else f_res
           p_p = p_res[0] if isinstance(p_res, tuple) else p_res
           p_gq = gq_res[0] if isinstance(gq_res, tuple) else gq_res
           p_h = h_res[0] if isinstance(h_res, tuple) else h_res


           bad_indices: set = set()


           def extract_positions(data) -> list:
               positions: list = []
               text = None
               if isinstance(data, tuple) and len(data) > 1:
                   text = data[1]
               elif isinstance(data, str):
                   text = data
               if text and isinstance(text, str):
                   for line in text.split('\n'):
                       if not line.strip():
                           continue
                       seq_str = line.split(':')[1] if ':' in line else line
                       clean_seq = clean_regex.sub('', seq_str).upper()
                       if len(clean_seq) >= 4:
                           search_start = 0
                           while True:
                               idx = full_transcript.find(clean_seq, search_start)
                               if idx == -1:
                                   break
                               cds_idx = idx - len(rbs_utr)
                               start_codon = max(0, cds_idx // 3)
                               end_codon = min(peptide_len, (cds_idx + len(clean_seq)) // 3 + 1)
                               if end_codon > start_codon:
                                   positions.extend(range(start_codon, end_codon))
                               search_start = idx + 1
               return positions


           codon_actually_failed = not p_c
           if codon_actually_failed and isinstance(c_res, tuple) and len(c_res) >= 4:
               diversity = c_res[1]
               rare_count = c_res[2]
               reported_cai = c_res[3]
               coding = [c for c in test_codons if c not in ('TAA', 'TGA', 'TAG')]
               real_cai = self.compute_cai(coding)
               if diversity >= 0.5 and rare_count <= 3 and reported_cai == 0.0 and real_cai >= 0.2:
                   codon_actually_failed = False


           if codon_actually_failed:
               coding_codons = test_codons[:-1]
               for i, codon in enumerate(coding_codons):
                   if codon in self.codon_checker.rare_codons:
                       bad_indices.add(i)
               for i, codon in enumerate(coding_codons):
                   freq = self.codon_checker.codon_frequencies.get(codon, 0.01)
                   if freq < 0.10:
                       bad_indices.add(i)
               unique_count = len(set(coding_codons))
               if unique_count < 31:
                   counts = Counter(coding_codons)
                   for codon, cnt in counts.most_common():
                       if cnt > 1:
                           indices_of_codon = [i for i, c in enumerate(coding_codons) if c == codon]
                           for idx in indices_of_codon[1:]:
                               bad_indices.add(idx)
                           if len(bad_indices) >= (31 - unique_count):
                               break


               if not bad_indices:
                   scored = [(i, self.codon_checker.codon_frequencies.get(c, 0.01))
                             for i, c in enumerate(coding_codons)]
                   scored.sort(key=lambda x: x[1])
                   n_target = max(3, len(scored) // 5)
                   for i, _ in scored[:n_target]:
                       bad_indices.add(i)


           if not p_h:
               chunk_size, overlap = 50, 25
               for i in range(0, len(full_transcript) - chunk_size + 1, overlap):
                   chunk = full_transcript[i:i + chunk_size]
                   count, _ = hairpin_counter(chunk, 3, 4, 9)
                   if count > 1:
                       cds_start = i - len(rbs_utr)
                       start_codon = max(0, cds_start // 3)
                       end_codon = min(peptide_len, (cds_start + chunk_size) // 3 + 1)
                       if end_codon > start_codon:
                           bad_indices.update(range(start_codon, end_codon))


           if not p_f:
               bad_indices.update(extract_positions(f_res))
           if not p_p:
               bad_indices.update(extract_positions(p_res))


           if not p_gq:
               gq_seq = gq_res[1] if isinstance(gq_res, tuple) and len(gq_res) > 1 else ""
               if gq_seq:
                   search_start = 0
                   while True:
                       idx = full_transcript.find(gq_seq, search_start)
                       if idx == -1:
                           break
                       cds_idx = idx - len(rbs_utr)
                       start_codon = max(0, cds_idx // 3)
                       end_codon = min(peptide_len, (cds_idx + len(gq_seq)) // 3 + 1)
                       if end_codon > start_codon:
                           bad_indices.update(range(start_codon, end_codon))
                       search_start = idx + 1


           return [i for i in bad_indices if 0 <= i < peptide_len]


       # --- Optimization loop ---
       best_errors, best_cai, passed, diags = evaluate(best_codons)
       if passed:
           return make_transcript(best_codons, "initial")


       assert diags is not None


       best_bad_indices = get_bad_indices(best_codons, diags)
       plateau_count = 0
       PLATEAU_LIMIT = 200
       MAX_ITERATIONS = 20000


       ever_best_codons = best_codons[:]
       ever_best_errors = best_errors


       for attempt in range(MAX_ITERATIONS):
           if time.time() - start_time > TIME_LIMIT:
               break


           test_codons = best_codons[:]


           h_res = diags[4]
           hairpin_failed = not (h_res[0] if isinstance(h_res, tuple) else h_res)


           c_res_current = diags[0]
           codon_failed = not (c_res_current[0] if isinstance(c_res_current, tuple) else c_res_current)


           if codon_failed and isinstance(c_res_current, tuple) and len(c_res_current) >= 4:
               diversity = c_res_current[1]
               rare_count = c_res_current[2]
               reported_cai = c_res_current[3]
               coding = [c for c in test_codons if c not in ('TAA', 'TGA', 'TAG')]
               real_cai = self.compute_cai(coding)
               if diversity >= 0.5 and rare_count <= 3 and reported_cai == 0.0 and real_cai >= 0.2:
                   codon_failed = False


           diversity_is_issue = False
           if codon_failed and isinstance(c_res_current, tuple) and len(c_res_current) >= 2:
               diversity = c_res_current[1]
               if diversity < 0.5:
                   diversity_is_issue = True


           if best_bad_indices:
               if hairpin_failed:
                   max_burst = min(5, len(best_bad_indices))
                   num_mutations = random.randint(2, max(2, max_burst))
               elif codon_failed:
                   max_burst = min(3, len(best_bad_indices))
                   num_mutations = random.randint(1, max(1, max_burst))
               else:
                   max_burst = min(3, len(best_bad_indices))
                   num_mutations = random.randint(1, max(1, max_burst))
               target_indices = random.sample(best_bad_indices, min(num_mutations, len(best_bad_indices)))
           else:
               target_indices = [random.randint(0, peptide_len - 1)]


           valid_mutation = False
           for idx in target_indices:
               if hairpin_failed:
                   available = [c for c in safe_mutation_map[idx] if c != best_codons[idx]]
                   if not available:
                       continue


                   codon_pos = len(rbs_utr) + idx * 3
                   in_bad_region = False
                   full_seq = rbs_utr + ''.join(test_codons)
                   for w in range(max(0, codon_pos - 49), min(len(full_seq) - 9, codon_pos + 3), 25):
                       chunk = full_seq[w:w + 50]
                       if len(chunk) >= 10:
                           cnt, _ = hairpin_counter(chunk, 3, 4, 9)
                           if cnt > 1:
                               in_bad_region = True
                               break


                   if in_bad_region:
                       scores: dict = {}
                       for cand in available:
                           trial = test_codons[:idx] + [cand] + test_codons[idx + 1:]
                           trial_seq = rbs_utr + ''.join(trial)
                           total = 0
                           for w in range(max(0, codon_pos - 49), min(len(trial_seq) - 9, codon_pos + 50), 25):
                               chunk = trial_seq[w:w + 50]
                               if len(chunk) >= 10:
                                   cnt, _ = hairpin_counter(chunk, 3, 4, 9)
                                   total += cnt
                           scores[cand] = total
                       best_score = min(scores.values())
                       top = [c for c, s in scores.items() if s == best_score]
                       test_codons[idx] = random.choice(top)
                   else:
                       test_codons[idx] = random.choice(available)


               elif codon_failed:
                   available = [c for c in full_mutation_map[idx] if c != best_codons[idx]]
                   if not available:
                       continue


                   coding = test_codons[:-1]
                   unique_count = len(set(coding))


                   if diversity_is_issue and unique_count < 31:
                       used = set(coding)
                       unused = [c for c in available if c not in used]
                       if unused:
                           test_codons[idx] = random.choice(unused)
                       else:
                           available.sort(
                               key=lambda c: self.codon_checker.codon_frequencies.get(c, 0.01),
                               reverse=True
                           )
                           test_codons[idx] = available[0]
                   else:
                       available.sort(
                           key=lambda c: self.codon_checker.codon_frequencies.get(c, 0.01),
                           reverse=True
                       )
                       if random.random() < 0.8:
                           test_codons[idx] = available[0]
                       else:
                           test_codons[idx] = random.choice(available)
               else:
                   available = [c for c in safe_mutation_map[idx] if c != best_codons[idx]]
                   if not available:
                       continue
                   weights = [self.codon_checker.codon_frequencies.get(c, 0.01) for c in available]
                   total_w = sum(weights)
                   if total_w > 0:
                       test_codons[idx] = random.choices(available, weights=weights, k=1)[0]
                   else:
                       test_codons[idx] = random.choice(available)
               valid_mutation = True


           if not valid_mutation:
               continue


           test_errors, test_cai, test_passed, test_diags = evaluate(test_codons, best_errors)


           if test_passed:
               return make_transcript(test_codons, "main_loop")


           accept = False
           if test_errors < best_errors:
               accept = True
           elif test_errors == best_errors:
               lateral_prob = max(0.1, 1.0 - attempt / MAX_ITERATIONS)
               if random.random() < lateral_prob:
                   accept = True
                   if test_cai < best_cai * 0.95:
                       accept = False


           if accept:
               plateau_count = 0
               best_codons = test_codons
               best_errors = test_errors
               best_cai = test_cai
               diags = test_diags
               assert diags is not None
               best_bad_indices = get_bad_indices(best_codons, diags)
               if best_errors < ever_best_errors:
                   ever_best_errors = best_errors
                   ever_best_codons = best_codons[:]
           else:
               plateau_count += 1


           if plateau_count >= PLATEAU_LIMIT:
               if time.time() - start_time > TIME_LIMIT:
                   break


               if best_errors <= 2 and best_bad_indices:
                   improved = False
                   for sweep_idx in list(best_bad_indices):
                       if time.time() - start_time > TIME_LIMIT:
                           break
                       for candidate in full_mutation_map.get(sweep_idx, safe_mutation_map[sweep_idx]):
                           if candidate == best_codons[sweep_idx]:
                               continue
                           sweep_codons = best_codons[:]
                           sweep_codons[sweep_idx] = candidate
                           s_err, s_cai, s_passed, s_diags = evaluate(sweep_codons)
                           if s_passed:
                               return make_transcript(sweep_codons, "sweep")
                           if s_err < best_errors:
                               best_codons = sweep_codons
                               best_errors = s_err
                               best_cai = s_cai
                               diags = s_diags
                               assert diags is not None
                               best_bad_indices = get_bad_indices(best_codons, diags)
                               plateau_count = 0
                               improved = True
                               if best_errors < ever_best_errors:
                                   ever_best_errors = best_errors
                                   ever_best_codons = best_codons[:]
                               break
                       if improved:
                           break


               if plateau_count >= PLATEAU_LIMIT:
                   best_codons = self.smart_restart(peptide)
                   best_codons.append("TAA")
                   best_errors, best_cai, passed, diags = evaluate(best_codons)
                   assert diags is not None
                   plateau_count = 0
                   if passed:
                       return make_transcript(best_codons, "restart")
                   best_bad_indices = get_bad_indices(best_codons, diags)
                   if best_errors < ever_best_errors:
                       ever_best_errors = best_errors
                       ever_best_codons = best_codons[:]


      
       # Exhausted iterations or time — pick the best available solution
       # Exhausted iterations or time — pick the best available solution
       final_errors, final_cai, final_passed, final_diags = evaluate(ever_best_codons)
       if final_passed:
           return make_transcript(ever_best_codons, "fallback_passed")


       curr_errors, curr_cai, curr_passed, curr_diags = evaluate(best_codons)
       if curr_passed:
           return make_transcript(best_codons, "fallback_curr_passed")


       # Return whichever has fewer errors
       if curr_errors <= final_errors:
           winner = best_codons
           w_errors = curr_errors
           w_diags = curr_diags
           w_label = "fallback_curr"
       else:
           winner = ever_best_codons
           w_errors = final_errors
           w_diags = final_diags
           w_label = "fallback_ever"


       # Last-ditch: try to fix remaining issues with a targeted sweep
       if w_errors > 0 and w_errors <= 2 and w_diags is not None:
           bad = get_bad_indices(winner, w_diags)
           if not bad:
               # extract_positions failed — brute force find the forbidden seq
               f_res = w_diags[1]
               if isinstance(f_res, tuple) and len(f_res) > 1 and f_res[1]:
                   forbidden_seq = f_res[1]
                   full_t = rbs_utr + ''.join(winner)
                   idx = full_t.find(forbidden_seq)
                   if idx >= 0:
                       cds_idx = idx - len(rbs_utr)
                       start_c = max(0, cds_idx // 3)
                       end_c = min(peptide_len, (cds_idx + len(forbidden_seq)) // 3 + 1)
                       bad = list(range(start_c, end_c))


           for sweep_idx in bad:
               if time.time() - start_time > TIME_LIMIT + 10:
                   break
               for candidate in full_mutation_map.get(sweep_idx, safe_mutation_map.get(sweep_idx, [])):
                   if candidate == winner[sweep_idx]:
                       continue
                   trial = winner[:]
                   trial[sweep_idx] = candidate
                   t_err, t_cai, t_passed, t_diags = evaluate(trial)
                   if t_passed:
                       return make_transcript(trial, "lastditch_passed")
                   if t_err < w_errors:
                       winner = trial
                       w_errors = t_err
                       w_diags = t_diags
                       w_label = "lastditch_improved"
                       break


       if w_errors > 0 and w_diags is not None:
           c_res, f_res, p_res, gq_res, h_res = w_diags
           coding = [c for c in winner if c not in ('TAA', 'TGA', 'TAG')]
           real_cai = self.compute_cai(coding)
           bad = get_bad_indices(winner, w_diags)
           print(f"\n--- DIAGNOSTIC FOR {peptide[:10]}... ---")
           print(f"Best errors: {w_errors}")
           print(f"Codon:       {c_res}")
           print(f"Real CAI:    {real_cai}")
           print(f"Forbidden:   {f_res}")
           print(f"Bad indices: {bad[:20]}")


       return make_transcript(winner, w_label)

