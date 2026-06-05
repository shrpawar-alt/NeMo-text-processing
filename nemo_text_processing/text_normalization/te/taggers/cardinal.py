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
 
from nemo_text_processing.text_normalization.te.graph_utils import GraphFst 
from nemo_text_processing.text_normalization.te.utils import get_abs_path 
 
 
class CardinalFst(GraphFst): 
    """ 
    Classifies cardinal numbers, e.g.  5  ->  cardinal { integer: "ఐదు" } 
    For compound numbers (21-99), it handles the pattern:  <tens> <units>, e.g.  42 ->  cardinal { integer: "నలభై రెండు" }
    For telugu digits, it uses the following mapping: e.g. ౧ -> cardinal { integer: "ఒకటి" }
    """ 
 
    def __init__(self, deterministic: bool = True): 
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic) 
 
        # Load the three data files as transducers  (number -> word) 
        digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")) 
        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")) 
        teens_and_ties = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv"))
        tens = pynini.string_file(get_abs_path("/data/numbers/tens.tsv"))

        # Predefined numbers (0-9, 10, 11 - 19, 20, 30, 40, 50, 60, 70, 80, 90)
        predefined = digit | zero | teens_and_ties

        # Two-digit compound (21-29, 31-39,...91-99)
        # For tens digit
        first_digit = pynini.union("2", "3", "4", "5", "6", "7", "8", "9")
    
        # Convert the tens value to tens word using teens_and_ties 
        tens_words = first_digit @ tens

        # For units digit
        unit_digit = pynini.union("1", "2", "3", "4", "5", "6", "7", "8", "9")

        # Convert the unit digit to unit word using predefined
        unit_words = unit_digit @ predefined

        # Compound pattern - tens_words + " " + unit_words
        compound = tens_words + pynutil.insert(" ") + unit_words
        
        graph = predefined | compound
        graph = graph.optimize()

        # Wrap in token field
        final_graph = pynutil.insert('integer: "') + graph + pynutil.insert('"')
 
        # add_tokens() turns it into:   cardinal { integer: "<word>" } 
        final_graph = self.add_tokens(final_graph) 
        self.fst = final_graph.optimize() 
