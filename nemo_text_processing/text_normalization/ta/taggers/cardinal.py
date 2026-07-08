

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
   The highest unit used is கோடி.
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

       single_digit = digit | zero
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

       # HUNDREDS GRAPH (100-999)

       # exact 100 → நூறு 
       exact_hundred = pynutil.add_weight(pynini.union(
           pynini.cross("100", "நூறு"),
           pynini.cross("௧௦௦", "நூறு"),
       ).optimize(), -1.0)

       # hundred prefixes: 200-900
       fused_hundred_prefix = pynini.union(
           pynini.cross("2", "இருநூற்று"),
           pynini.cross("3", "முந்நூற்று"),
           pynini.cross("4", "நானூற்று"),
           pynini.cross("5", "ஐநூற்று"),
           pynini.cross("6", "அறுநூற்று"),
           pynini.cross("7", "எழுநூற்று"),
           pynini.cross("8", "எட்டுநூற்று"),
           pynini.cross("9", "ஒன்பதுநூற்று"),
           pynini.cross("௨", "இருநூற்று"),
           pynini.cross("௩", "முந்நூற்று"),
           pynini.cross("௪", "நானூற்று"),
           pynini.cross("௫", "ஐநூற்று"),
           pynini.cross("௬", "அறுநூற்று"),
           pynini.cross("௭", "எழுநூற்று"),
           pynini.cross("௮", "எட்டுநூற்று"),
           pynini.cross("௯", "ஒன்பதுநூற்று"),
       ).optimize()

       graph_hundreds = hundred_ta
       # 101-109
       graph_hundreds |= (
           (pynini.cross("1", "நூற்று") | pynini.cross("௧", "நூற்று"))
           + zero_delete + pynutil.insert(" ") + single_digit
       )
       # 110-199
       graph_hundreds |= (
           (pynini.cross("1", "நூற்று") | pynini.cross("௧", "நூற்று"))
           + pynutil.insert(" ") + teens_ties
       )
       # 201-209 ... 901-909
       graph_hundreds |= fused_hundred_prefix + zero_delete + pynutil.insert(" ") + single_digit
       # 210-999
       graph_hundreds |= fused_hundred_prefix + pynutil.insert(" ") + teens_ties
       graph_hundreds = graph_hundreds.optimize()
       self.graph_hundreds = graph_hundreds

       # THOUSANDS GRAPH (1000-9999)
       # 1000-1999 → ஆயிரம்/ஆயிரத்து
       one_k_exact = pynini.cross("1", "ஆயிரம்") | pynini.cross("௧", "ஆயிரம்")
       one_k_tail  = pynini.cross("1", "ஆயிரத்து") | pynini.cross("௧", "ஆயிரத்து")
       # 2000-9000 
       fused_thousands_exact = pynini.union(
           pynini.cross("2", "இரண்டாயிரம்"),
           pynini.cross("3", "மூவாயிரம்"),
           pynini.cross("4", "நான்காயிரம்"),
           pynini.cross("5", "ஐந்தாயிரம்"),
           pynini.cross("6", "ஆறாயிரம்"),
           pynini.cross("7", "ஏழாயிரம்"),
           pynini.cross("8", "எட்டாயிரம்"),
           pynini.cross("9", "ஒன்பதாயிரம்"),
           pynini.cross("௨", "இரண்டாயிரம்"),
           pynini.cross("௩", "மூவாயிரம்"),
           pynini.cross("௪", "நான்காயிரம்"),
           pynini.cross("௫", "ஐந்தாயிரம்"),
           pynini.cross("௬", "ஆறாயிரம்"),
           pynini.cross("௭", "ஏழாயிரம்"),
           pynini.cross("௮", "எட்டாயிரம்"),
           pynini.cross("௯", "ஒன்பதாயிரம்"),
       ).optimize()

       fused_thousands_prefix = pynini.union(
           pynini.cross("2", "இரண்டாயிரத்து"),
           pynini.cross("3", "மூவாயிரத்து"),
           pynini.cross("4", "நான்காயிரத்து"),
           pynini.cross("5", "ஐந்தாயிரத்து"),
           pynini.cross("6", "ஆறாயிரத்து"),
           pynini.cross("7", "ஏழாயிரத்து"),
           pynini.cross("8", "எட்டாயிரத்து"),
           pynini.cross("9", "ஒன்பதாயிரத்து"),
           pynini.cross("௨", "இரண்டாயிரத்து"),
           pynini.cross("௩", "மூவாயிரத்து"),
           pynini.cross("௪", "நான்காயிரத்து"),
           pynini.cross("௫", "ஐந்தாயிரத்து"),
           pynini.cross("௬", "ஆறாயிரத்து"),
           pynini.cross("௭", "ஏழாயிரத்து"),
           pynini.cross("௮", "எட்டாயிரத்து"),
           pynini.cross("௯", "ஒன்பதாயிரத்து"),
       ).optimize()

       graph_thousands = one_k_exact + (zero_delete ** 3)
       graph_thousands |= one_k_tail + (zero_delete ** 2) + pynutil.insert(" ") + single_digit
       graph_thousands |= one_k_tail + zero_delete + pynutil.insert(" ") + teens_ties
       graph_thousands |= one_k_tail + pynutil.insert(" ") + graph_hundreds
       graph_thousands |= fused_thousands_exact + (zero_delete ** 3)
       graph_thousands |= fused_thousands_prefix + (zero_delete ** 2) + pynutil.insert(" ") + single_digit
       graph_thousands |= fused_thousands_prefix + zero_delete + pynutil.insert(" ") + teens_ties
       graph_thousands |= fused_thousands_prefix + pynutil.insert(" ") + graph_hundreds
       graph_thousands = graph_thousands.optimize()
       self.graph_thousands = graph_thousands

       # TEN-THOUSANDS GRAPH (10000-99999)
         # 10000-19999 
       fused_ten_k_exact = pynini.union(
           pynini.cross("௧௩", "பதிமூன்றுஆயிரம்"),
           pynini.cross("௧௨", "பன்னிரண்டுஆயிரம்"),
           pynini.cross("௧௧", "பதினொன்றுஆயிரம்"),
           pynini.cross("௪௫", "நாற்பத்தைந்தாயிரம்"),
           pynini.cross("௫௫", "ஐம்பத்தைந்தாயிரம்"),
       ).optimize()

       fused_ten_k_prefix = pynini.union(
           pynini.cross("௧௩", "பதிமூன்றுஆயிரத்து"),
           pynini.cross("௧௨", "பன்னிரண்டுஆயிரத்து"),
           pynini.cross("௧௧", "பதினொன்றுஆயிரத்து"),
           pynini.cross("௪௫", "நாற்பத்தைந்தாயிரத்து"),
           pynini.cross("௫௫", "ஐம்பத்தைந்தாயிரத்து"),
       ).optimize()

       suffix_ten_k_exact = pynutil.insert("ஆயிரம்")
       suffix_ten_k_tail  = pynutil.insert("ஆயிரத்து")

       graph_ten_thousands = fused_ten_k_exact + (zero_delete ** 3)
       graph_ten_thousands |= fused_ten_k_prefix + (zero_delete ** 2) + pynutil.insert(" ") + single_digit
       graph_ten_thousands |= fused_ten_k_prefix + zero_delete + pynutil.insert(" ") + teens_ties
       graph_ten_thousands |= fused_ten_k_prefix + pynutil.insert(" ") + graph_hundreds
       graph_ten_thousands |= create_graph_suffix(teens_and_ties, suffix_ten_k_exact, 3)
       graph_ten_thousands |= create_larger_number_graph(teens_and_ties, suffix_ten_k_tail, 2, single_digit)
       graph_ten_thousands |= create_larger_number_graph(teens_and_ties, suffix_ten_k_tail, 1, teens_ties)
       graph_ten_thousands |= create_larger_number_graph(teens_and_ties, suffix_ten_k_tail, 0, graph_hundreds)
       graph_ten_thousands = graph_ten_thousands.optimize()
       self.graph_ten_thousands = graph_ten_thousands

       # LAKHS GRAPH (100000-9999999)
       # 1 lakh → ஒரு லட்சம்
       suffix_lakhs      = pynutil.insert(" லட்சம்")
       suffix_lakhs_tail = pynutil.insert(" லட்சத்து")

       digit_one_as_oru = pynini.union(
           pynini.cross("1", "ஒரு"),
           pynini.cross("௧", "ஒரு"),
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
       graph_ten_lakhs = graph_ten_lakhs.optimize()
       self.graph_ten_lakhs = graph_ten_lakhs

       # CRORES GRAPH — கோடி 
       # 1 crore → ஒரு கோடி / ஒரு கோடியே
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
       graph_ten_crores = graph_ten_crores.optimize()

       # hundreds of crores: 100-999 கோடி
       graph_hundreds_of_crores  = create_graph_suffix(exact_hundred, suffix_crores, 7)
       graph_hundreds_of_crores |= create_graph_suffix(graph_hundreds, suffix_crores, 7)
       graph_hundreds_of_crores |= create_larger_number_graph(exact_hundred, suffix_crores_tail, 6, single_digit)
       graph_hundreds_of_crores |= create_larger_number_graph(exact_hundred, suffix_crores_tail, 5, teens_ties)
       graph_hundreds_of_crores |= create_larger_number_graph(exact_hundred, suffix_crores_tail, 4, graph_hundreds)
       graph_hundreds_of_crores |= create_larger_number_graph(exact_hundred, suffix_crores_tail, 3, graph_thousands)
       graph_hundreds_of_crores |= create_larger_number_graph(exact_hundred, suffix_crores_tail, 2, graph_ten_thousands)
       graph_hundreds_of_crores |= create_larger_number_graph(exact_hundred, suffix_crores_tail, 1, graph_lakhs)
       graph_hundreds_of_crores |= create_larger_number_graph(exact_hundred, suffix_crores_tail, 0, graph_ten_lakhs)
       graph_hundreds_of_crores |= create_larger_number_graph(graph_hundreds, suffix_crores_tail, 6, single_digit)
       graph_hundreds_of_crores |= create_larger_number_graph(graph_hundreds, suffix_crores_tail, 5, teens_ties)
       graph_hundreds_of_crores |= create_larger_number_graph(graph_hundreds, suffix_crores_tail, 4, graph_hundreds)
       graph_hundreds_of_crores |= create_larger_number_graph(graph_hundreds, suffix_crores_tail, 3, graph_thousands)
       graph_hundreds_of_crores |= create_larger_number_graph(graph_hundreds, suffix_crores_tail, 2, graph_ten_thousands)
       graph_hundreds_of_crores |= create_larger_number_graph(graph_hundreds, suffix_crores_tail, 1, graph_lakhs)
       graph_hundreds_of_crores |= create_larger_number_graph(graph_hundreds, suffix_crores_tail, 0, graph_ten_lakhs)
       graph_hundreds_of_crores = graph_hundreds_of_crores.optimize()

       # thousands of crores: 1000-9999 கோடி
       graph_thousands_of_crores  = create_graph_suffix(graph_thousands, suffix_crores, 7)
       graph_thousands_of_crores |= create_larger_number_graph(graph_thousands, suffix_crores_tail, 6, single_digit)
       graph_thousands_of_crores |= create_larger_number_graph(graph_thousands, suffix_crores_tail, 5, teens_ties)
       graph_thousands_of_crores |= create_larger_number_graph(graph_thousands, suffix_crores_tail, 4, graph_hundreds)
       graph_thousands_of_crores |= create_larger_number_graph(graph_thousands, suffix_crores_tail, 3, graph_thousands)
       graph_thousands_of_crores |= create_larger_number_graph(graph_thousands, suffix_crores_tail, 2, graph_ten_thousands)
       graph_thousands_of_crores |= create_larger_number_graph(graph_thousands, suffix_crores_tail, 1, graph_lakhs)
       graph_thousands_of_crores |= create_larger_number_graph(graph_thousands, suffix_crores_tail, 0, graph_ten_lakhs)
       graph_thousands_of_crores = graph_thousands_of_crores.optimize()

       # ten-thousands of crores
       graph_ten_thousands_of_crores  = create_graph_suffix(graph_ten_thousands, suffix_crores, 7)
       graph_ten_thousands_of_crores |= create_larger_number_graph(graph_ten_thousands, suffix_crores_tail, 6, single_digit)
       graph_ten_thousands_of_crores |= create_larger_number_graph(graph_ten_thousands, suffix_crores_tail, 5, teens_ties)
       graph_ten_thousands_of_crores |= create_larger_number_graph(graph_ten_thousands, suffix_crores_tail, 4, graph_hundreds)
       graph_ten_thousands_of_crores |= create_larger_number_graph(graph_ten_thousands, suffix_crores_tail, 3, graph_thousands)
       graph_ten_thousands_of_crores |= create_larger_number_graph(graph_ten_thousands, suffix_crores_tail, 2, graph_ten_thousands)
       graph_ten_thousands_of_crores |= create_larger_number_graph(graph_ten_thousands, suffix_crores_tail, 1, graph_lakhs)
       graph_ten_thousands_of_crores |= create_larger_number_graph(graph_ten_thousands, suffix_crores_tail, 0, graph_ten_lakhs)
       graph_ten_thousands_of_crores = graph_ten_thousands_of_crores.optimize()

       # lakhs of crores
       graph_lakhs_of_crores  = create_graph_suffix(graph_lakhs, suffix_crores, 7)
       graph_lakhs_of_crores |= create_larger_number_graph(graph_lakhs, suffix_crores_tail, 6, single_digit)
       graph_lakhs_of_crores |= create_larger_number_graph(graph_lakhs, suffix_crores_tail, 5, teens_ties)
       graph_lakhs_of_crores |= create_larger_number_graph(graph_lakhs, suffix_crores_tail, 4, graph_hundreds)
       graph_lakhs_of_crores |= create_larger_number_graph(graph_lakhs, suffix_crores_tail, 3, graph_thousands)
       graph_lakhs_of_crores |= create_larger_number_graph(graph_lakhs, suffix_crores_tail, 2, graph_ten_thousands)
       graph_lakhs_of_crores |= create_larger_number_graph(graph_lakhs, suffix_crores_tail, 1, graph_lakhs)
       graph_lakhs_of_crores |= create_larger_number_graph(graph_lakhs, suffix_crores_tail, 0, graph_ten_lakhs)
       graph_lakhs_of_crores = graph_lakhs_of_crores.optimize()

       graph_ten_lakhs_of_crores  = create_graph_suffix(graph_ten_lakhs, suffix_crores, 7)
       graph_ten_lakhs_of_crores |= create_larger_number_graph(graph_ten_lakhs, suffix_crores_tail, 6, single_digit)
       graph_ten_lakhs_of_crores |= create_larger_number_graph(graph_ten_lakhs, suffix_crores_tail, 5, teens_ties)
       graph_ten_lakhs_of_crores |= create_larger_number_graph(graph_ten_lakhs, suffix_crores_tail, 4, graph_hundreds)
       graph_ten_lakhs_of_crores |= create_larger_number_graph(graph_ten_lakhs, suffix_crores_tail, 3, graph_thousands)
       graph_ten_lakhs_of_crores |= create_larger_number_graph(graph_ten_lakhs, suffix_crores_tail, 2, graph_ten_thousands)
       graph_ten_lakhs_of_crores |= create_larger_number_graph(graph_ten_lakhs, suffix_crores_tail, 1, graph_lakhs)
       graph_ten_lakhs_of_crores |= create_larger_number_graph(graph_ten_lakhs, suffix_crores_tail, 0, graph_ten_lakhs)
       graph_ten_lakhs_of_crores = graph_ten_lakhs_of_crores.optimize()

       # crores of crores
       graph_crores_of_crores  = create_graph_suffix(graph_crores, suffix_crores, 7)
       graph_crores_of_crores |= create_larger_number_graph(graph_crores, suffix_crores_tail, 6, single_digit)
       graph_crores_of_crores |= create_larger_number_graph(graph_crores, suffix_crores_tail, 5, teens_ties)
       graph_crores_of_crores |= create_larger_number_graph(graph_crores, suffix_crores_tail, 4, graph_hundreds)
       graph_crores_of_crores |= create_larger_number_graph(graph_crores, suffix_crores_tail, 3, graph_thousands)
       graph_crores_of_crores |= create_larger_number_graph(graph_crores, suffix_crores_tail, 2, graph_ten_thousands)
       graph_crores_of_crores |= create_larger_number_graph(graph_crores, suffix_crores_tail, 1, graph_lakhs)
       graph_crores_of_crores |= create_larger_number_graph(graph_crores, suffix_crores_tail, 0, graph_ten_lakhs)
       graph_crores_of_crores = graph_crores_of_crores.optimize()

       graph_ten_crores_of_crores  = create_graph_suffix(graph_ten_crores, suffix_crores, 7)
       graph_ten_crores_of_crores |= create_larger_number_graph(graph_ten_crores, suffix_crores_tail, 6, single_digit)
       graph_ten_crores_of_crores |= create_larger_number_graph(graph_ten_crores, suffix_crores_tail, 5, teens_ties)
       graph_ten_crores_of_crores |= create_larger_number_graph(graph_ten_crores, suffix_crores_tail, 4, graph_hundreds)
       graph_ten_crores_of_crores |= create_larger_number_graph(graph_ten_crores, suffix_crores_tail, 3, graph_thousands)
       graph_ten_crores_of_crores |= create_larger_number_graph(graph_ten_crores, suffix_crores_tail, 2, graph_ten_thousands)
       graph_ten_crores_of_crores |= create_larger_number_graph(graph_ten_crores, suffix_crores_tail, 1, graph_lakhs)
       graph_ten_crores_of_crores |= create_larger_number_graph(graph_ten_crores, suffix_crores_tail, 0, graph_ten_lakhs)
       graph_ten_crores_of_crores = graph_ten_crores_of_crores.optimize()

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
           | graph_hundreds_of_crores
           | graph_thousands_of_crores
           | graph_ten_thousands_of_crores
           | graph_lakhs_of_crores
           | graph_ten_lakhs_of_crores
           | graph_crores_of_crores
           | graph_ten_crores_of_crores
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
