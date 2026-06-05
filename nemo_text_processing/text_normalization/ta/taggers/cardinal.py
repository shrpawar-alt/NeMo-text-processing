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
 
from nemo_text_processing.text_normalization.ta.graph_utils import GraphFst 
from nemo_text_processing.text_normalization.ta.utils import get_abs_path 
 
 
class CardinalFst(GraphFst):
    """ 
    Finite state transducer for classifying cardinals, e.g.
        5 -> cardinal { integer: "ஐந்து" }
    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """ 
 
    def __init__(self, deterministic: bool = True): 
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic) 
 
        # Load the three data files as transducers  (number -> word) 
        digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")) 
        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")) 
        teens_and_ties = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv")) 
      
        graph = digit | zero | teens_and_ties            
        graph = graph.optimize() 
 
        final_graph = pynutil.insert('integer: "') + graph + pynutil.insert('"') 
 
        # add_tokens() turns it into:   cardinal { integer: "<word>" } 
        final_graph = self.add_tokens(final_graph) 
        self.fst = final_graph.optimize() 