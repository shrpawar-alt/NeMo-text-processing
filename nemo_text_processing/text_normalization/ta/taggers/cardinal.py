# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pynini
from pynini.lib import pynutil
from nemo_text_processing.text_normalization.ta.graph_utils import (
   NEMO_ALL_DIGIT,
   NEMO_ALL_ZERO,
   GraphFst,
   insert_space,
)
from nemo_text_processing.text_normalization.ta.utils import get_abs_path

class CardinalFst(GraphFst):
   """
   Finite state transducer for classifying cardinals, e.g.
       -23 -> cardinal { negative: "true"  integer: "இருபத்து மூன்று" }
   Args:
       deterministic: if True will provide a single transduction option,
           for False multiple transduction are generated (used for audio-based normalization)
   """
   def __init__(self, deterministic: bool = True, lm: bool = False):
       super().__init__(name="cardinal", kind="classify", deterministic=deterministic)
       digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
       zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
       hundred_ta = pynini.string_file(get_abs_path("data/numbers/hundred_ta.tsv"))
       teens_ties_hi = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv"))
       teens_ties_en = pynini.string_file(get_abs_path("data/numbers/teens_and_ties_en.tsv"))
       teens_ties = pynini.union(teens_ties_hi, teens_ties_en)
       teens_and_ties = pynutil.add_weight(teens_ties, -0.1)
       self.digit = digit
       self.zero = zero
       self.hundred_ta = hundred_ta
       self.teens_and_ties = teens_and_ties
       # single_digit early — covers ASCII + Tamil digits + zero
       single_digit = digit | zero
       # Single digit graph for digit-by-digit reading
       single_digit_graph = digit | zero
       self.single_digits_graph = single_digit_graph + pynini.closure(insert_space + single_digit_graph)
       zero_delete = pynutil.add_weight(pynutil.delete(NEMO_ALL_ZERO | pynini.accep("௦")), -0.1)
       def create_graph_suffix(digit_graph, suffix, zeros_counts):
           z = pynutil.add_weight(pynutil.delete(NEMO_ALL_ZERO | pynini.accep("௦")), -0.1)
           if zeros_counts == 0:
               return digit_graph + suffix
           return digit_graph + (z ** zeros_counts) + suffix
       def create_larger_number_graph(digit_graph, suffix, zeros_counts, sub_graph):
           ins = pynutil.insert(" ")
           z = pynutil.add_weight(pynutil.delete(NEMO_ALL_ZERO | pynini.accep("௦")), -0.1)
           if zeros_counts == 0:
               return digit_graph + suffix + ins + sub_graph
           return digit_graph + suffix + (z ** zeros_counts) + ins + sub_graph
       
       # HUNDREDS GRAPH
       # Use hundred_ta.tsv for fused exact hundreds (நூற்று, இருநூற்று, ...
       # எட்டுநூற்று, எழுநூற்று, நானூற்று etc.)
       # For 101-199: நூற்று + space + remainder
       # For 201-999: fused prefix (no space) + space + remainder
       # Fused hundred prefix for composites (201-999 non-exact)
       fused_hundred_prefix = pynini.union(
           pynini.cross("2", "இருநூற்று"),
           pynini.cross("3", "முந்நூற்று"),
           pynini.cross("4", "நானூற்று"),
           pynini.cross("5", "ஐந்நூற்று"),
           pynini.cross("6", "அறுநூற்று"),
           pynini.cross("7", "எழுநூற்று"),
           pynini.cross("8", "எட்டுநூற்று"),
           pynini.cross("9", "ஒன்பதுநூற்று"),
           pynini.cross("௨", "இருநூற்று"),
           pynini.cross("௩", "முந்நூற்று"),
           pynini.cross("௪", "நானூற்று"),
           pynini.cross("௫", "ஐந்நூற்று"),
           pynini.cross("௬", "அறுநூற்று"),
           pynini.cross("௭", "எழுநூற்று"),
           pynini.cross("௮", "எட்டுநூற்று"),
           pynini.cross("௯", "ஒன்பதுநூற்று"),
       ).optimize()
       # Start with fused exact forms from hundred_ta.tsv
       exact_hundred = pynutil.add_weight(pynini.union(
           pynini.cross("100", "நூறு"),
           pynini.cross("௧௦௦", "நூறு"),
        ).optimize(), -1.0)
       graph_hundreds = hundred_ta
       # 101-109 (ASCII and Tamil digits)
       graph_hundreds |= (
           (pynini.cross("1", "நூற்று") | pynini.cross("௧", "நூற்று"))
           + zero_delete
           + pynutil.insert(" ")
           + single_digit
       )
       # 110-119
       graph_hundreds |= (
           (pynini.cross("1", "நூற்று") | pynini.cross("௧", "நூற்று"))
           + pynutil.insert(" ")
           + teens_ties
       )
       # 201-209, 301-309 ... 901-909
       graph_hundreds |= fused_hundred_prefix + zero_delete + pynutil.insert(" ") + single_digit
       # 210-219 ... 910-919
       graph_hundreds |= fused_hundred_prefix + pynutil.insert(" ") + teens_ties
       # 220-299 ... 920-999 (two-digit tie/tens tail already in teens_ties)
       # teens_ties handles 20,21,...99 so this covers the remaining composites
       graph_hundreds = graph_hundreds.optimize()
       self.graph_hundreds = graph_hundreds
       # THOUSANDS GRAPH
       # 1000      → ஆயிரம்  (exact, no leading ஒன்று)
       # 1001-1099 → ஆயிரத்து + space + remainder
       # 1100-1999 → ஆயிரத்து + space + graph_hundreds
       # 2000-9000 → fused exact (இரண்டாயிரம், மூன்றாயிரம் ...)
       # 2001-9999 → fused prefix (ஐந்தாயிரத்து ...) + space + remainder
       one_k_exact = pynini.cross("1", "ஆயிரம்") | pynini.cross("௧", "ஆயிரம்")
       one_k_tail  = pynini.cross("1", "ஆயிரத்து") | pynini.cross("௧", "ஆயிரத்து")
       fused_thousands_exact = pynini.union(
           pynini.cross("2", "இரண்டாயிரம்"),
           pynini.cross("3", "மூன்றாயிரம்"),
           pynini.cross("4", "நான்காயிரம்"),
           pynini.cross("5", "ஐந்தாயிரம்"),
           pynini.cross("6", "ஆறாயிரம்"),
           pynini.cross("7", "ஏழாயிரம்"),
           pynini.cross("8", "எட்டாயிரம்"),
           pynini.cross("9", "ஒன்பதாயிரம்"),
           pynini.cross("௨", "இரண்டாயிரம்"),
           pynini.cross("௩", "மூன்றாயிரம்"),
           pynini.cross("௪", "நான்காயிரம்"),
           pynini.cross("௫", "ஐந்தாயிரம்"),
           pynini.cross("௬", "ஆறாயிரம்"),
           pynini.cross("௭", "ஏழாயிரம்"),
           pynini.cross("௮", "எட்டாயிரம்"),
           pynini.cross("௯", "ஒன்பதாயிரம்"),
       ).optimize()
       fused_thousands_prefix = pynini.union(
           pynini.cross("2", "இரண்டாயிரத்து"),
           pynini.cross("3", "மூன்றாயிரத்து"),
           pynini.cross("4", "நான்காயிரத்து"),
           pynini.cross("5", "ஐந்தாயிரத்து"),
           pynini.cross("6", "ஆறாயிரத்து"),
           pynini.cross("7", "ஏழாயிரத்து"),
           pynini.cross("8", "எட்டாயிரத்து"),
           pynini.cross("9", "ஒன்பதாயிரத்து"),
           pynini.cross("௨", "இரண்டாயிரத்து"),
           pynini.cross("௩", "மூன்றாயிரத்து"),
           pynini.cross("௪", "நான்காயிரத்து"),
           pynini.cross("௫", "ஐந்தாயிரத்து"),
           pynini.cross("௬", "ஆறாயிரத்து"),
           pynini.cross("௭", "ஏழாயிரத்து"),
           pynini.cross("௮", "எட்டாயிரத்து"),
           pynini.cross("௯", "ஒன்பதாயிரத்து"),
       ).optimize()
       # 1000 exact
       graph_thousands = one_k_exact + (zero_delete ** 3)
       # 1001-1009
       graph_thousands |= one_k_tail + (zero_delete ** 2) + pynutil.insert(" ") + single_digit
       # 1010-1099
       graph_thousands |= one_k_tail + zero_delete + pynutil.insert(" ") + teens_ties
       # 1100-1999
       graph_thousands |= one_k_tail + pynutil.insert(" ") + graph_hundreds
       # 2000-9000 exact
       graph_thousands |= fused_thousands_exact + (zero_delete ** 3)
       # 2001-9009
       graph_thousands |= fused_thousands_prefix + (zero_delete ** 2) + pynutil.insert(" ") + single_digit
       # 2010-9099
       graph_thousands |= fused_thousands_prefix + zero_delete + pynutil.insert(" ") + teens_ties
       # 2100-9999
       graph_thousands |= fused_thousands_prefix + pynutil.insert(" ") + graph_hundreds
       graph_thousands = graph_thousands.optimize()
       self.graph_thousands = graph_thousands
       # TEN-THOUSANDS GRAPH
       # e.g. 55000 → ஐம்பத்தைந்தாயிரம்,  55199 → ஐம்பத்தைந்தாயிரத்து ...
       # teens_and_ties already covers 10-99; we fuse them with ஆயிரம்/ஆயிரத்து
       suffix_ten_k_exact = pynutil.insert("ஆயிரம்")
       suffix_ten_k_tail  = pynutil.insert("ஆயிரத்து")
       # exact ten-thousands (X0000 where X is a two-digit teens/ties)
       graph_ten_thousands = create_graph_suffix(teens_and_ties, suffix_ten_k_exact, 3)
       # +single digit tail
       graph_ten_thousands |= create_larger_number_graph(teens_and_ties, suffix_ten_k_tail, 2, single_digit)
       # +teens tail
       graph_ten_thousands |= create_larger_number_graph(teens_and_ties, suffix_ten_k_tail, 1, teens_ties)
       # +hundreds tail
       graph_ten_thousands |= create_larger_number_graph(teens_and_ties, suffix_ten_k_tail, 0, graph_hundreds)
       graph_ten_thousands = graph_ten_thousands.optimize()
       self.graph_ten_thousands = graph_ten_thousands
       
       # LAKHS — இலட்சம் / இலட்சத்து
       suffix_lakhs      = pynutil.insert(" இலட்சம்")
       suffix_lakhs_tail = pynutil.insert(" இலட்சத்து")
       digit_one_as_oru = pynini.union(
           pynini.cross("1", "ஒன்று"),
           pynini.cross("௧", "ஒன்று"),
           pynini.cross("2", "இரண்டு"),
           pynini.cross("3", "மூன்று"),
           pynini.cross("4", "நான்கு"),
           pynini.cross("5", "ஐந்து"),
           pynini.cross("6", "ஆறு"),
           pynini.cross("7", "ஏழு"),
           pynini.cross("8", "எட்டு"),
           pynini.cross("9", "ஒன்பது"),
           pynini.cross("௨", "இரண்டு"),
           pynini.cross("௩", "மூன்று"),
           pynini.cross("௪", "நான்கு"),
           pynini.cross("௫", "ஐந்து"),
           pynini.cross("௬", "ஆறு"),
           pynini.cross("௭", "ஏழு"),
           pynini.cross("௮", "எட்டு"),
           pynini.cross("௯", "ஒன்பது"),
       ).optimize()
       graph_lakhs  = create_graph_suffix(digit_one_as_oru, suffix_lakhs, 5)
       graph_lakhs |= create_larger_number_graph(digit_one_as_oru, suffix_lakhs_tail, 4, single_digit)
       graph_lakhs |= create_larger_number_graph(digit_one_as_oru, suffix_lakhs_tail, 3, teens_ties)
       graph_lakhs |= create_larger_number_graph(digit_one_as_oru, suffix_lakhs_tail, 2, graph_hundreds)
       graph_lakhs |= create_larger_number_graph(digit_one_as_oru, suffix_lakhs_tail, 1, graph_thousands)
       graph_lakhs |= create_larger_number_graph(digit_one_as_oru, suffix_lakhs_tail, 0, graph_ten_thousands)
       graph_lakhs  = graph_lakhs.optimize()
       self.graph_lakhs = graph_lakhs
       graph_ten_lakhs  = create_graph_suffix(teens_and_ties, suffix_lakhs, 5)
       graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs_tail, 4, single_digit)
       graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs_tail, 3, teens_ties)
       graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs_tail, 2, graph_hundreds)
       graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs_tail, 1, graph_thousands)
       graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs_tail, 0, graph_ten_thousands)
       graph_ten_lakhs.optimize()
       self.graph_ten_lakhs = graph_ten_lakhs

       # CRORES — கோடி / கோடியே
       suffix_crores      = pynutil.insert(" கோடி")
       suffix_crores_tail = pynutil.insert(" கோடியே")
       graph_crores  = create_graph_suffix(digit_one_as_oru, suffix_crores, 7)
       graph_crores |= create_larger_number_graph(digit_one_as_oru, suffix_crores_tail, 6, single_digit)
       graph_crores |= create_larger_number_graph(digit_one_as_oru, suffix_crores_tail, 5, teens_ties)
       graph_crores |= create_larger_number_graph(digit_one_as_oru, suffix_crores_tail, 4, graph_hundreds)
       graph_crores |= create_larger_number_graph(digit_one_as_oru, suffix_crores_tail, 3, graph_thousands)
       graph_crores |= create_larger_number_graph(digit_one_as_oru, suffix_crores_tail, 2, graph_ten_thousands)
       graph_crores |= create_larger_number_graph(digit_one_as_oru, suffix_crores_tail, 1, graph_lakhs)
       graph_crores |= create_larger_number_graph(digit_one_as_oru, suffix_crores_tail, 0, graph_ten_lakhs)
       graph_crores  = graph_crores.optimize()
       graph_ten_crores  = create_graph_suffix(teens_and_ties, suffix_crores, 7)
       graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores_tail, 6, single_digit)
       graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores_tail, 5, teens_ties)
       graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores_tail, 4, graph_hundreds)
       graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores_tail, 3, graph_thousands)
       graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores_tail, 2, graph_ten_thousands)
       graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores_tail, 1, graph_lakhs)
       graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores_tail, 0, graph_ten_lakhs)
       graph_ten_crores.optimize()

       # ARABS
       suffix_arabs = pynutil.insert(" அரபு")
       graph_arabs  = create_graph_suffix(digit_one_as_oru, suffix_arabs, 9)
       graph_arabs |= create_larger_number_graph(digit_one_as_oru, suffix_arabs, 8, single_digit)
       graph_arabs |= create_larger_number_graph(digit_one_as_oru, suffix_arabs, 7, teens_ties)
       graph_arabs |= create_larger_number_graph(digit_one_as_oru, suffix_arabs, 6, graph_hundreds)
       graph_arabs |= create_larger_number_graph(digit_one_as_oru, suffix_arabs, 5, graph_thousands)
       graph_arabs |= create_larger_number_graph(digit_one_as_oru, suffix_arabs, 4, graph_ten_thousands)
       graph_arabs |= create_larger_number_graph(digit_one_as_oru, suffix_arabs, 3, graph_lakhs)
       graph_arabs |= create_larger_number_graph(digit_one_as_oru, suffix_arabs, 2, graph_ten_lakhs)
       graph_arabs |= create_larger_number_graph(digit_one_as_oru, suffix_arabs, 1, graph_crores)
       graph_arabs |= create_larger_number_graph(digit_one_as_oru, suffix_arabs, 0, graph_ten_crores)
       graph_arabs.optimize()
       graph_ten_arabs  = create_graph_suffix(teens_and_ties, suffix_arabs, 9)
       graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 8, single_digit)
       graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 7, teens_ties)
       graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 6, graph_hundreds)
       graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 5, graph_thousands)
       graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 4, graph_ten_thousands)
       graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 3, graph_lakhs)
       graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 2, graph_ten_lakhs)
       graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 1, graph_crores)
       graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 0, graph_ten_crores)
       graph_ten_arabs.optimize()
       
       # KHARABS
       suffix_kharabs = pynutil.insert(" கரபு")
       graph_kharabs  = create_graph_suffix(digit_one_as_oru, suffix_kharabs, 11)
       graph_kharabs |= create_larger_number_graph(digit_one_as_oru, suffix_kharabs, 10, single_digit)
       graph_kharabs |= create_larger_number_graph(digit_one_as_oru, suffix_kharabs, 9, teens_ties)
       graph_kharabs |= create_larger_number_graph(digit_one_as_oru, suffix_kharabs, 8, graph_hundreds)
       graph_kharabs |= create_larger_number_graph(digit_one_as_oru, suffix_kharabs, 7, graph_thousands)
       graph_kharabs |= create_larger_number_graph(digit_one_as_oru, suffix_kharabs, 6, graph_ten_thousands)
       graph_kharabs |= create_larger_number_graph(digit_one_as_oru, suffix_kharabs, 5, graph_lakhs)
       graph_kharabs |= create_larger_number_graph(digit_one_as_oru, suffix_kharabs, 4, graph_ten_lakhs)
       graph_kharabs |= create_larger_number_graph(digit_one_as_oru, suffix_kharabs, 3, graph_crores)
       graph_kharabs |= create_larger_number_graph(digit_one_as_oru, suffix_kharabs, 2, graph_ten_crores)
       graph_kharabs |= create_larger_number_graph(digit_one_as_oru, suffix_kharabs, 1, graph_arabs)
       graph_kharabs |= create_larger_number_graph(digit_one_as_oru, suffix_kharabs, 0, graph_ten_arabs)
       graph_kharabs.optimize()
       graph_ten_kharabs  = create_graph_suffix(teens_and_ties, suffix_kharabs, 11)
       graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 10, single_digit)
       graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 9, teens_ties)
       graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 8, graph_hundreds)
       graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 7, graph_thousands)
       graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 6, graph_ten_thousands)
       graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 5, graph_lakhs)
       graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 4, graph_ten_lakhs)
       graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 3, graph_crores)
       graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 2, graph_ten_crores)
       graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 1, graph_arabs)
       graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 0, graph_ten_arabs)
       graph_ten_kharabs.optimize()

       # NILS
       suffix_nils = pynutil.insert(" நீல்")
       graph_nils  = create_graph_suffix(digit_one_as_oru, suffix_nils, 13)
       graph_nils |= create_larger_number_graph(digit_one_as_oru, suffix_nils, 12, single_digit)
       graph_nils |= create_larger_number_graph(digit_one_as_oru, suffix_nils, 11, teens_ties)
       graph_nils |= create_larger_number_graph(digit_one_as_oru, suffix_nils, 10, graph_hundreds)
       graph_nils |= create_larger_number_graph(digit_one_as_oru, suffix_nils, 9, graph_thousands)
       graph_nils |= create_larger_number_graph(digit_one_as_oru, suffix_nils, 8, graph_ten_thousands)
       graph_nils |= create_larger_number_graph(digit_one_as_oru, suffix_nils, 7, graph_lakhs)
       graph_nils |= create_larger_number_graph(digit_one_as_oru, suffix_nils, 6, graph_ten_lakhs)
       graph_nils |= create_larger_number_graph(digit_one_as_oru, suffix_nils, 5, graph_crores)
       graph_nils |= create_larger_number_graph(digit_one_as_oru, suffix_nils, 4, graph_ten_crores)
       graph_nils |= create_larger_number_graph(digit_one_as_oru, suffix_nils, 3, graph_arabs)
       graph_nils |= create_larger_number_graph(digit_one_as_oru, suffix_nils, 2, graph_ten_arabs)
       graph_nils |= create_larger_number_graph(digit_one_as_oru, suffix_nils, 1, graph_kharabs)
       graph_nils |= create_larger_number_graph(digit_one_as_oru, suffix_nils, 0, graph_ten_kharabs)
       graph_nils.optimize()
       graph_ten_nils  = create_graph_suffix(teens_and_ties, suffix_nils, 13)
       graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 12, single_digit)
       graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 11, teens_ties)
       graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 10, graph_hundreds)
       graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 9, graph_thousands)
       graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 8, graph_ten_thousands)
       graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 7, graph_lakhs)
       graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 6, graph_ten_lakhs)
       graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 5, graph_crores)
       graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 4, graph_ten_crores)
       graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 3, graph_arabs)
       graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 2, graph_ten_arabs)
       graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 1, graph_kharabs)
       graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 0, graph_ten_kharabs)
       graph_ten_nils.optimize()
    
       # PADMAS
       suffix_padmas = pynutil.insert(" பத்மம்")
       graph_padmas  = create_graph_suffix(digit_one_as_oru, suffix_padmas, 15)
       graph_padmas |= create_larger_number_graph(digit_one_as_oru, suffix_padmas, 14, single_digit)
       graph_padmas |= create_larger_number_graph(digit_one_as_oru, suffix_padmas, 13, teens_ties)
       graph_padmas |= create_larger_number_graph(digit_one_as_oru, suffix_padmas, 12, graph_hundreds)
       graph_padmas |= create_larger_number_graph(digit_one_as_oru, suffix_padmas, 11, graph_thousands)
       graph_padmas |= create_larger_number_graph(digit_one_as_oru, suffix_padmas, 10, graph_ten_thousands)
       graph_padmas |= create_larger_number_graph(digit_one_as_oru, suffix_padmas, 9, graph_lakhs)
       graph_padmas |= create_larger_number_graph(digit_one_as_oru, suffix_padmas, 8, graph_ten_lakhs)
       graph_padmas |= create_larger_number_graph(digit_one_as_oru, suffix_padmas, 7, graph_crores)
       graph_padmas |= create_larger_number_graph(digit_one_as_oru, suffix_padmas, 6, graph_ten_crores)
       graph_padmas |= create_larger_number_graph(digit_one_as_oru, suffix_padmas, 5, graph_arabs)
       graph_padmas |= create_larger_number_graph(digit_one_as_oru, suffix_padmas, 4, graph_ten_arabs)
       graph_padmas |= create_larger_number_graph(digit_one_as_oru, suffix_padmas, 3, graph_kharabs)
       graph_padmas |= create_larger_number_graph(digit_one_as_oru, suffix_padmas, 2, graph_ten_kharabs)
       graph_padmas |= create_larger_number_graph(digit_one_as_oru, suffix_padmas, 1, graph_nils)
       graph_padmas |= create_larger_number_graph(digit_one_as_oru, suffix_padmas, 0, graph_ten_nils)
       graph_padmas.optimize()
       graph_ten_padmas  = create_graph_suffix(teens_and_ties, suffix_padmas, 15)
       graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 14, single_digit)
       graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 13, teens_ties)
       graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 12, graph_hundreds)
       graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 11, graph_thousands)
       graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 10, graph_ten_thousands)
       graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 9, graph_lakhs)
       graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 8, graph_ten_lakhs)
       graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 7, graph_crores)
       graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 6, graph_ten_crores)
       graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 5, graph_arabs)
       graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 4, graph_ten_arabs)
       graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 3, graph_kharabs)
       graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 2, graph_ten_kharabs)
       graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 1, graph_nils)
       graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 0, graph_ten_nils)
       graph_ten_padmas.optimize()
       
       # SHANKHS
       suffix_shankhs = pynutil.insert(" சங்கம்")
       graph_shankhs  = create_graph_suffix(digit_one_as_oru, suffix_shankhs, 17)
       graph_shankhs |= create_larger_number_graph(digit_one_as_oru, suffix_shankhs, 16, single_digit)
       graph_shankhs |= create_larger_number_graph(digit_one_as_oru, suffix_shankhs, 15, teens_ties)
       graph_shankhs |= create_larger_number_graph(digit_one_as_oru, suffix_shankhs, 14, graph_hundreds)
       graph_shankhs |= create_larger_number_graph(digit_one_as_oru, suffix_shankhs, 13, graph_thousands)
       graph_shankhs |= create_larger_number_graph(digit_one_as_oru, suffix_shankhs, 12, graph_ten_thousands)
       graph_shankhs |= create_larger_number_graph(digit_one_as_oru, suffix_shankhs, 11, graph_lakhs)
       graph_shankhs |= create_larger_number_graph(digit_one_as_oru, suffix_shankhs, 10, graph_ten_lakhs)
       graph_shankhs |= create_larger_number_graph(digit_one_as_oru, suffix_shankhs, 9, graph_crores)
       graph_shankhs |= create_larger_number_graph(digit_one_as_oru, suffix_shankhs, 8, graph_ten_crores)
       graph_shankhs |= create_larger_number_graph(digit_one_as_oru, suffix_shankhs, 7, graph_arabs)
       graph_shankhs |= create_larger_number_graph(digit_one_as_oru, suffix_shankhs, 6, graph_ten_arabs)
       graph_shankhs |= create_larger_number_graph(digit_one_as_oru, suffix_shankhs, 5, graph_kharabs)
       graph_shankhs |= create_larger_number_graph(digit_one_as_oru, suffix_shankhs, 4, graph_ten_kharabs)
       graph_shankhs |= create_larger_number_graph(digit_one_as_oru, suffix_shankhs, 3, graph_nils)
       graph_shankhs |= create_larger_number_graph(digit_one_as_oru, suffix_shankhs, 2, graph_ten_nils)
       graph_shankhs |= create_larger_number_graph(digit_one_as_oru, suffix_shankhs, 1, graph_padmas)
       graph_shankhs |= create_larger_number_graph(digit_one_as_oru, suffix_shankhs, 0, graph_ten_padmas)
       graph_shankhs.optimize()
       graph_ten_shankhs  = create_graph_suffix(teens_and_ties, suffix_shankhs, 17)
       graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 16, single_digit)
       graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 15, teens_ties)
       graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 14, graph_hundreds)
       graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 13, graph_thousands)
       graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 12, graph_ten_thousands)
       graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 11, graph_lakhs)
       graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 10, graph_ten_lakhs)
       graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 9, graph_crores)
       graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 8, graph_ten_crores)
       graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 7, graph_arabs)
       graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 6, graph_ten_arabs)
       graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 5, graph_kharabs)
       graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 4, graph_ten_kharabs)
       graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 3, graph_nils)
       graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 2, graph_ten_nils)
       graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 1, graph_padmas)
       graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 0, graph_ten_padmas)
       graph_ten_shankhs.optimize()

       # LEADING ZERO and FINAL GRAPH
       graph_leading_zero = zero + insert_space + single_digit
       graph_leading_zero = pynutil.add_weight(graph_leading_zero, 0.5)
       graph_without_leading_zeros = (
           digit
           | zero
           | teens_and_ties
           | exact_hundred
           | graph_hundreds
           | graph_thousands
           | graph_ten_thousands
           | graph_lakhs
           | graph_ten_lakhs
           | graph_crores
           | graph_ten_crores
           | graph_arabs
           | graph_ten_arabs
           | graph_kharabs
           | graph_ten_kharabs
           | graph_nils
           | graph_ten_nils
           | graph_padmas
           | graph_ten_padmas
           | graph_shankhs
           | graph_ten_shankhs
       )
       self.graph_without_leading_zeros = graph_without_leading_zeros.optimize()
       cardinal_with_leading_zeros = pynini.compose(
           NEMO_ALL_ZERO + pynini.closure(NEMO_ALL_DIGIT), self.single_digits_graph
       )
       cardinal_with_leading_zeros = pynutil.add_weight(cardinal_with_leading_zeros, 0.5)
       final_graph = graph_without_leading_zeros | cardinal_with_leading_zeros
       optional_minus_graph = pynini.closure(
           pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1
       )
       self.final_graph = final_graph.optimize()
       final_graph = (
           optional_minus_graph
           + pynutil.insert("integer: \"")
           + self.final_graph
           + pynutil.insert("\"")
       )
       final_graph = self.add_tokens(final_graph)
       self.fst = final_graph