# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.hi.graph_utils import (
    ASTERISK,
    COMMA,
    DECIMAL_25,
    DECIMAL_75,
    HI_BY,
    HI_DEDH,
    HI_DHAI,
    HI_PAUNE,
    HI_PERIOD,
    HI_SADHE,
    HI_SAVVA,
    HYPHEN,
    LOWERCASE_X,
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_HI_DIGIT,
    NEMO_NOT_SPACE,
    NEMO_SIGMA,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    ONE_POINT_FIVE,
    PERIOD,
    POINT_FIVE,
    SLASH,
    TWO_POINT_FIVE,
    UPPERCASE_X,
    GraphFst,
    capitalized_input_graph,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.hi.utils import get_abs_path

digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
teens_ties_hi = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv"))
teens_ties_en = pynini.string_file(get_abs_path("data/numbers/teens_and_ties_en.tsv"))
teens_ties = pynini.union(teens_ties_hi, teens_ties_en)
teens_and_ties = pynutil.add_weight(teens_ties, -0.1)
# Shared Address Maps
zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
telephone_number = pynini.string_file(get_abs_path("data/telephone/number.tsv"))
states_map = pynini.string_file(get_abs_path("data/address/states.tsv"))
cities_map = pynini.string_file(get_abs_path("data/address/cities.tsv"))
special_characters_map = pynini.string_file(get_abs_path("data/address/special_characters.tsv"))
letters_map = pynini.string_file(get_abs_path("data/address/letters.tsv"))
context_map = pynini.string_file(get_abs_path("data/address/context.tsv"))


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure, suppletive aware, e.g.
        -१२kg -> measure { negative: "true" cardinal { integer: "बारह" } units: "किलोग्राम" }
        -१२.२kg -> measure { decimal { negative: "true"  integer_part: "बारह"  fractional_part: "दो"} units: "किलोग्राम" }
        मुंबई ८८४४०४ -> measure { units: "address" cardinal { integer: "मुंबई आठ आठ चार चार शून्य चार" } preserve_order: true }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def get_structured_address_graph(self, cardinal: GraphFst, ordinal: GraphFst, input_case: str):
        """
        Minimal address tagger for state/city + pincode patterns only.

        Examples:
            "मुंबई ८८४४०४" -> "मुंबई आठ आठ चार चार शून्य चार"
            "गोवा १२३४५६" -> "गोवा एक दो तीन चार पाँच छह"
            "100 फीट रोड, चेन्नई" -> "एक सौ फीट रोड, चेन्नई"
        """
        # State/city keywords

        state_city_names = pynini.union(states_map, cities_map).optimize()

        # Digit mappings (shared maps loaded once at module level)
        num_token = (digit | zero | telephone_number).optimize()

        # Pincode (6 digits) -> always digit-by-digit (length >= 4)
        pincode = (num_token + pynini.closure(insert_space + num_token, 5, 5)).optimize()

        # Street number: 1-3 digits read as cardinal, 4+ digits read digit-by-digit
        num_1to3 = (cardinal.digit | cardinal.zero | cardinal.teens_and_ties | cardinal.graph_hundreds).optimize()
        num_4plus = (num_token + pynini.closure(insert_space + num_token, 3)).optimize()
        street_num = (num_1to3 | num_4plus).optimize()

        # Text: words with trailing separator (comma? + space)
        any_digit = pynini.union(NEMO_HI_DIGIT, NEMO_DIGIT).optimize()
        punctuation = pynini.union(COMMA, PERIOD, HI_PERIOD).optimize()
        word_char = pynini.difference(NEMO_NOT_SPACE, pynini.union(any_digit, punctuation)).optimize()
        word = pynini.closure(word_char, 1)

        # Separator: optional comma followed by mandatory space
        sep = pynini.closure(pynini.accep(COMMA), 0, 1) + pynini.accep(NEMO_SPACE)
        word_with_sep = word + sep
        # Consume inline address numbers using the same 1-3 (cardinal) / 4+ (digit-by-digit) rule
        num_with_sep = street_num + sep
        text = pynini.closure(pynini.union(word_with_sep, num_with_sep), 0, 5).optimize()

        # Pattern: [street_num + sep]? text state/city [space pincode]
        pattern = (
            pynini.closure(street_num + sep, 0, 1)
            + text
            + state_city_names
            + pynini.closure(pynini.accep(NEMO_SPACE) + pincode, 0, 1)
        ).optimize()

        graph = (
            pynutil.insert('units: "address" cardinal { integer: "')
            + pattern
            + pynutil.insert('" } preserve_order: true')
        )
        return pynutil.add_weight(graph, 1.0).optimize()

    def get_address_graph(self, cardinal: GraphFst, ordinal: GraphFst, input_case: str):
        """
        Address tagger that fires when address context keywords are present.

        Examples:
            "७०० ओक स्ट्रीट" -> "सात सौ ओक स्ट्रीट"
            "६६-४ पार्क रोड" -> "छियासठ हाइफ़न चार पार्क रोड"
            "593988" (6-digit pincode) -> "पाँच नौ तीन नौ आठ आठ"
            "32A नाज़ प्लाज़ा" -> "बत्तीस ए नाज़ प्लाज़ा"
        """
        # Strip internal weights from ordinal graph so a small outer weight suffices
        ordinal_graph = pynini.arcmap(ordinal.graph, map_type="rmweight").optimize()
        # Alphanumeric to word mappings (digits, special characters, telephone digits)
        char_to_word = (digit | zero | special_characters_map | telephone_number).optimize()
        letter_to_word = capitalized_input_graph(letters_map)
        # Identity acceptor for keywords (Devanagari/English) to prevent unintended rewrites/transliteration
        address_keywords = pynini.project(
            capitalized_input_graph(context_map),
            "input",
        ).optimize()

        # Alphanumeric processing: treat digits, letters, and -/ as convertible tokens
        single_digit = pynini.union(NEMO_DIGIT, NEMO_HI_DIGIT).optimize()
        special_chars = pynini.union(HYPHEN, SLASH).optimize()
        single_letter = pynini.project(letter_to_word, "input").optimize()
        convertible_char = pynini.union(single_digit, special_chars, single_letter)
        non_space_char = pynini.difference(
            NEMO_CHAR, pynini.union(NEMO_WHITE_SPACE, convertible_char, pynini.accep(COMMA))
        ).optimize()

        # Token processors with weights: prefer ordinals; English words are left untouched
        # Delete space before comma to avoid Sparrowhawk "sil" issue
        comma_processor = pynutil.add_weight(delete_space + pynini.accep(COMMA), 0.0)
        # Slight preference (-0.5) to ensure ordinals beat alphanumeric code splits
        ordinal_processor = pynutil.add_weight(insert_space + ordinal_graph, -0.5)
        # Pass English words (2+ letters) through unchanged; single letters go to letter_processor
        latin_word = single_letter + pynini.closure(single_letter, 1)
        english_word_processor = pynutil.add_weight(insert_space + latin_word, 0.1)
        letter_processor = pynutil.add_weight(insert_space + pynini.compose(single_letter, letter_to_word), 0.5)
        # Transliterate separators ("-"/"/"); digits are handled separately by number_run_processor
        special_char_processor = pynutil.add_weight(insert_space + pynini.compose(special_chars, char_to_word), 0.0)
        other_word_processor = pynutil.add_weight(insert_space + pynini.closure(non_space_char, 1), 0.1)

        # --- Alphanumeric codes (letter+digit): letters -> Devanagari, digits via cross-class rule ---
        code_letter = pynini.compose(single_letter, letter_to_word).optimize()
        code_letters = code_letter + pynini.closure(insert_space + code_letter)
        code_num_1to3 = (cardinal.digit | cardinal.zero | cardinal.teens_and_ties | cardinal.graph_hundreds).optimize()
        code_num_4plus = pynini.compose(
            single_digit**4 + pynini.closure(single_digit), cardinal.single_digits_graph
        ).optimize()
        code_num = (code_num_1to3 | code_num_4plus).optimize()
        code_seg = (code_letters | code_num).optimize()
        code_delim = pynini.union(HYPHEN, SLASH, NEMO_SPACE).optimize()
        code_core = (code_seg + pynini.closure(code_delim + code_seg, 1)).optimize()
        # Insert spaces at letter<->digit boundaries for glued codes (e.g., "F16" -> "F 16")
        insert_sp_alpha_digit = pynini.cdrewrite(insert_space, single_letter, single_digit, NEMO_SIGMA)
        insert_sp_digit_alpha = pynini.cdrewrite(insert_space, single_digit, single_letter, NEMO_SIGMA)
        code_space_inserter = pynini.compose(insert_sp_alpha_digit, insert_sp_digit_alpha).optimize()
        glued_code = pynini.compose(code_space_inserter, code_core).optimize()
        code_transliterate = pynini.union(code_core, glued_code).optimize()
        # Restrict to tokens containing BOTH a letter and a digit to prevent fragmented parsing
        code_char = pynini.union(single_letter, single_digit, pynini.accep(HYPHEN), pynini.accep(SLASH))
        has_latin_letter = pynini.closure(code_char) + single_letter + pynini.closure(code_char)
        has_digit = pynini.closure(code_char) + single_digit + pynini.closure(code_char)
        code_only = pynini.intersect(
            pynini.intersect(pynini.closure(code_char, 1), has_latin_letter), has_digit
        ).optimize()
        code_transliterate = pynini.compose(code_only, code_transliterate).optimize()
        # Strip internal weights; standalone codes naturally beat split alternatives without outer weights
        code_transliterate = pynini.arcmap(code_transliterate, map_type="rmweight").optimize()
        code_processor = insert_space + code_transliterate

        # Pure numeric runs: 1-3 digits read as cardinal, 4+ digits read digit-by-digit
        number_run = pynini.compose(pynini.closure(single_digit, 1), code_num).optimize()
        number_run_processor = pynutil.add_weight(insert_space + number_run, 0.5)

        token_processor = (
            ordinal_processor
            | english_word_processor
            | code_processor
            | number_run_processor
            | letter_processor
            | special_char_processor
            | pynini.accep(NEMO_SPACE)
            | comma_processor
            | other_word_processor
        ).optimize()
        full_string_processor = pynini.closure(token_processor, 1).optimize()

        # Window-based context matching around address keywords for robust detection
        word_boundary = pynini.union(
            NEMO_WHITE_SPACE, pynini.accep(COMMA), pynini.accep(HI_PERIOD), pynini.accep(PERIOD)
        ).optimize()
        non_boundary_char = pynini.difference(NEMO_CHAR, word_boundary)
        word = pynini.closure(non_boundary_char, 1).optimize()
        word_with_boundary = word + pynini.closure(word_boundary)
        window = pynini.closure(word_with_boundary, 0, 5).optimize()
        boundary = pynini.closure(word_boundary, 1).optimize()
        input_pattern = pynini.union(
            address_keywords + boundary + window,
            window + boundary + address_keywords + pynini.closure(boundary + window, 0, 1),
        ).optimize()
        address_graph = pynini.compose(input_pattern, full_string_processor).optimize()
        graph = (
            pynutil.insert('units: "address" cardinal { integer: "')
            + address_graph
            + pynutil.insert('" } preserve_order: true')
        )
        return pynutil.add_weight(graph, 1.05).optimize()

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, ordinal: GraphFst, input_case: str):
        super().__init__(name="measure", kind="classify")

        cardinal_graph = (
            cardinal.zero
            | cardinal.digit
            | cardinal.teens_and_ties
            | cardinal.graph_hundreds
            | cardinal.graph_thousands
            | cardinal.graph_ten_thousands
            | cardinal.graph_lakhs
            | cardinal.graph_ten_lakhs
        )
        point = pynutil.delete(".")
        decimal_integers = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        decimal_graph = decimal_integers + point + insert_space + decimal.graph_fractional
        unit_graph = pynini.string_file(get_abs_path("data/measure/unit.tsv"))

        # Year unit variants for formal/informal handling
        year_informal = pynini.string_map([("yr", "साल")])
        year_formal = pynini.string_file(get_abs_path("data/measure/unit_year_formal.tsv"))

        # All units EXCEPT year
        unit_inputs_except_yr = pynini.difference(pynini.project(unit_graph, "input"), pynini.accep("yr"))
        unit_graph_no_year = pynini.compose(unit_inputs_except_yr, unit_graph)

        # Load quarterly units from separate files: map (FST) and list (FSA)
        quarterly_units_map = pynini.string_file(get_abs_path("data/measure/quarterly_units_map.tsv"))
        quarterly_units_list = pynini.string_file(get_abs_path("data/measure/quarterly_units_list.tsv"))
        quarterly_units_graph = pynini.union(quarterly_units_map, quarterly_units_list)

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + insert_space,
            0,
            1,
        )

        # Define the quarterly measurements - support both Devanagari and Arabic digits
        quarter = pynini.union(
            pynini.cross(POINT_FIVE, HI_SADHE),
            pynini.cross(ONE_POINT_FIVE, HI_DEDH),
            pynini.cross(TWO_POINT_FIVE, HI_DHAI),
        )
        quarter_graph = pynutil.insert("integer_part: \"") + quarter + pynutil.insert("\"")

        # Define the unit handling
        unit = (
            pynutil.insert(NEMO_SPACE)
            + pynutil.insert("units: \"")
            + unit_graph_no_year
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )
        units = (
            pynutil.insert(NEMO_SPACE)
            + pynutil.insert("units: \"")
            + quarterly_units_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )

        # Year-specific unit wrappers
        unit_year_informal = (
            pynutil.insert(NEMO_SPACE)
            + pynutil.insert("units: \"")
            + year_informal
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )
        unit_year_formal = (
            pynutil.insert(NEMO_SPACE)
            + pynutil.insert("units: \"")
            + year_formal
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )

        # Cardinal >= 1000 -> formal year (वर्ष)
        # Use graph_without_leading_zeros which covers all number ranges (thousands to shankhs)
        cardinal_large = cardinal.graph_without_leading_zeros

        # Cardinal < 1000 -> informal year (साल)
        cardinal_small = cardinal.zero | cardinal.digit | cardinal.teens_and_ties | cardinal.graph_hundreds

        symbol_graph = pynini.string_map(
            [
                (LOWERCASE_X, HI_BY),
                (UPPERCASE_X, HI_BY),
                (ASTERISK, HI_BY),
            ]
        )

        graph_decimal = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal_graph
            + pynutil.insert(" }")
            + delete_space
            + unit
        )

        # Support both Devanagari and Arabic digits for dedh/dhai patterns
        dedh_dhai = pynini.union(
            pynini.cross(ONE_POINT_FIVE, HI_DEDH),
            pynini.cross(TWO_POINT_FIVE, HI_DHAI),
        )
        dedh_dhai_graph = pynutil.insert("integer: \"") + dedh_dhai + pynutil.insert("\"")

        # Support both Devanagari and Arabic digits for savva pattern
        savva_numbers = cardinal_graph + pynini.cross(DECIMAL_25, "")
        savva_graph = (
            pynutil.insert("integer: \"")
            + pynutil.insert(HI_SAVVA)
            + pynutil.insert(NEMO_SPACE)
            + savva_numbers
            + pynutil.insert("\"")
        )

        # Support both Devanagari and Arabic digits for sadhe pattern
        sadhe_numbers = cardinal_graph + pynini.cross(POINT_FIVE, "")
        sadhe_graph = (
            pynutil.insert("integer: \"")
            + pynutil.insert(HI_SADHE)
            + pynutil.insert(NEMO_SPACE)
            + sadhe_numbers
            + pynutil.insert("\"")
        )

        paune = pynini.string_file(get_abs_path("data/whitelist/paune_mappings.tsv"))
        # Support both Devanagari and Arabic digits for paune pattern
        paune_numbers = paune + pynini.cross(DECIMAL_75, "")
        paune_graph = (
            pynutil.insert("integer: \"")
            + pynutil.insert(HI_PAUNE)
            + pynutil.insert(NEMO_SPACE)
            + paune_numbers
            + pynutil.insert("\"")
        )

        graph_dedh_dhai = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + dedh_dhai_graph
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
            + delete_space
            + units
        )

        graph_savva = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + savva_graph
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
            + delete_space
            + units
        )

        graph_sadhe = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + sadhe_graph
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
            + delete_space
            + units
        )

        graph_paune = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + paune_graph
            + pynutil.insert(" }")
            + delete_space
            + units
        )

        graph_cardinal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
            + delete_space
            + unit
        )

        # Large numbers (>=1000) + yr -> formal (वर्ष)
        graph_cardinal_year_formal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_large
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
            + delete_space
            + unit_year_formal
        )

        # Small numbers (<1000) + yr -> informal (साल)
        graph_cardinal_year_informal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_small
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
            + delete_space
            + unit_year_informal
        )

        # Regular decimals (e.g., 16.07) + yr -> formal (वर्ष)
        graph_decimal_year_formal = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal_graph
            + pynutil.insert(" }")
            + delete_space
            + unit_year_formal
        )

        # Handling cardinal clubbed with symbol as single token
        graph_exceptions = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("units: \"")
            + symbol_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("} }")
            + insert_space
            + pynutil.insert("tokens { cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + pynutil.insert("\"")
        )

        address_graph = self.get_address_graph(cardinal, ordinal, input_case)
        structured_address_graph = self.get_structured_address_graph(cardinal, ordinal, input_case)

        graph = (
            pynutil.add_weight(graph_decimal, 0.1)
            | pynutil.add_weight(graph_decimal_year_formal, 0.1)
            | pynutil.add_weight(graph_cardinal, 0.1)
            | pynutil.add_weight(graph_cardinal_year_formal, 0.1)
            | pynutil.add_weight(graph_cardinal_year_informal, -0.1)  # Higher priority for small numbers
            | pynutil.add_weight(graph_exceptions, 0.1)
            | pynutil.add_weight(graph_dedh_dhai, -0.2)
            | pynutil.add_weight(graph_savva, -0.1)
            | pynutil.add_weight(graph_sadhe, -0.1)
            | pynutil.add_weight(graph_paune, -0.5)
            | address_graph
            | structured_address_graph
        )
        self.graph = graph.optimize()

        final_graph = self.add_tokens(graph)
        self.fst = final_graph
