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
    INPUT_LOWER_CASED,
    LOWERCASE_X,
    MIN_NEG_WEIGHT,
    NEMO_ALL_DIGIT,
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_HI_DIGIT,
    NEMO_NOT_SPACE,
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
# Load both Hindi (Devanagari) and English (Arabic) number mappings
teens_ties_hi = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv"))
teens_ties_en = pynini.string_file(get_abs_path("data/numbers/teens_and_ties_en.tsv"))
teens_ties = pynini.union(teens_ties_hi, teens_ties_en)
teens_and_ties = pynutil.add_weight(teens_ties, -0.1)


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure, suppletive aware, e.g.
        -१२kg -> measure { negative: "true" cardinal { integer: "बारह" } units: "किलोग्राम" }
        -१२.२kg -> measure { decimal { negative: "true"  integer_part: "बारह"  fractional_part: "दो"} units: "किलोग्राम" }
        मुंबई ८८४४०४ -> measure { units: "address" cardinal { integer: "मुंबई आठ आठ चार चार शून्य चार" } preserve_order: true }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        ordinal: OrdinalFst
        input_case: str
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def get_structured_address_graph(self, states_graph: "pynini.Fst", cities_graph: "pynini.Fst"):
        """
        Minimal address tagger for state/city + pincode patterns only.

        Examples:
            "मुंबई ८८४४०४" -> "मुंबई आठ आठ चार चार शून्य चार"
            "गोवा १२३४५६" -> "गोवा एक दो तीन चार पाँच छह"
        """
        state_city_names = pynini.union(states_graph, cities_graph).optimize()

        # Digit mappings
        num_token = (
            digit
            | pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
            | pynini.string_file(get_abs_path("data/telephone/number.tsv"))
        ).optimize()

        # Pincode (6 digits)
        pincode = (num_token + pynini.closure(insert_space + num_token, 5, 5)).optimize()

        # Street number (1-4 digits)
        street_num = (num_token + pynini.closure(insert_space + num_token, 0, 3)).optimize()

        # Text: words with trailing separator
        any_digit = pynini.union(NEMO_HI_DIGIT, NEMO_DIGIT).optimize()
        punctuation = pynini.union(COMMA, PERIOD, HI_PERIOD).optimize()
        word_char = pynini.difference(NEMO_NOT_SPACE, pynini.union(any_digit, punctuation)).optimize()
        word = pynini.closure(word_char, 1)

        # Separator: optional comma followed by mandatory space
        sep = pynini.closure(pynini.accep(COMMA), 0, 1) + pynini.accep(NEMO_SPACE)
        word_with_sep = word + sep
        text = pynini.closure(word_with_sep, 0, 5).optimize()

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
        # Weight kept above get_address_graph so address graph takes priority when both can match.
        return pynutil.add_weight(graph, 2.0).optimize()

    def get_address_graph(
        self,
        cardinal: GraphFst,
        ordinal: GraphFst,
        input_case: str,
        address_keywords_hi: "pynini.Fst",
        states_graph: "pynini.Fst",
        cities_graph: "pynini.Fst",
        en_to_hi_map: "pynini.Fst",
    ):
        """
        Address tagger with natural Hindi verbalization for street numbers.
        1-3 digit numbers are verbalized as cardinals, 4+ digit numbers and
        PIN codes as digit-by-digit.

        Examples:
            "७०० ओक स्ट्रीट"  -> "सात सौ ओक स्ट्रीट"
            "६६-४ पार्क रोड"  -> "छियासठ हाइफ़न चार पार्क रोड"
            "14/3 मथुरा रोड"  -> "चौदह बटा तीन मथुरा रोड"
        """
        special_char_map = pynini.string_file(get_abs_path("data/address/special_characters.tsv"))
        single_digit_word = (cardinal.digit | cardinal.zero).optimize()
        single_digit_input = NEMO_ALL_DIGIT

        digit_run = (single_digit_word + pynini.closure(insert_space + single_digit_word)).optimize()

        num_1digit = pynini.compose(single_digit_input**1, (cardinal.digit | cardinal.zero)).optimize()
        num_2digit = pynini.compose(single_digit_input**2, cardinal.teens_and_ties).optimize()
        num_3digit = pynini.compose(single_digit_input**3, cardinal.graph_hundreds).optimize()

        num_as_cardinal = (
            pynutil.add_weight(num_3digit, -0.2)
            | pynutil.add_weight(num_2digit, -0.02)
            | pynutil.add_weight(num_1digit, MIN_NEG_WEIGHT)
        ).optimize()

        digit_run_4_plus = pynini.compose(
            single_digit_input**4 + pynini.closure(single_digit_input), digit_run
        ).optimize()

        any_street_num = (num_as_cardinal | pynutil.add_weight(digit_run_4_plus, -0.5)).optimize()

        # Only slash joins two numbers into one token; hyphen is handled by standalone_hyphen.
        slash_separator = pynini.compose(pynini.accep(SLASH), special_char_map)
        joined_num = (any_street_num + insert_space + slash_separator + insert_space + any_street_num).optimize()

        # Digit run maps to one token by length: 1-3 digits -> cardinal, 4+ -> digit-by-digit (covers PIN codes too).
        street_num_1_to_3 = num_as_cardinal
        street_num_4_plus = digit_run_4_plus

        ordinal_processor = pynutil.add_weight(insert_space + ordinal.graph, MIN_NEG_WEIGHT * 5)

        if input_case != INPUT_LOWER_CASED:
            en_to_hi_map = capitalized_input_graph(en_to_hi_map)
        english_word_processor = pynutil.add_weight(insert_space + en_to_hi_map, MIN_NEG_WEIGHT)

        letter_to_word = pynini.string_file(get_abs_path("data/address/letters.tsv"))
        letter_to_word = capitalized_input_graph(letter_to_word)
        single_letter = pynini.project(letter_to_word, "input").optimize()
        letter_processor = pynutil.add_weight(insert_space + pynini.compose(single_letter, letter_to_word), 0.5)

        special_chars = pynini.union(HYPHEN, SLASH).optimize()
        convertible_char = pynini.union(single_digit_input, special_chars, single_letter)
        non_space_char = pynini.difference(
            NEMO_CHAR, pynini.union(NEMO_WHITE_SPACE, convertible_char, pynini.accep(COMMA))
        ).optimize()

        # Letter right after a slash (e.g. "12/A").
        right_side_letter = (
            pynini.compose(single_letter, letter_to_word) | pynini.closure(non_space_char, 1)
        ).optimize()
        slash_word = pynini.compose(pynini.accep(SLASH), special_char_map)
        slashed_num_letter = (any_street_num + insert_space + slash_word + insert_space + right_side_letter).optimize()

        comma_processor = pynutil.add_weight(delete_space + pynini.accep(COMMA), 0.0)

        hyphen_word = pynini.compose(pynini.accep(HYPHEN), special_char_map)
        standalone_hyphen = hyphen_word

        other_word_processor = pynutil.add_weight(insert_space + pynini.closure(non_space_char, 1), 0.1)

        # Kept separate so full_string_processor can forbid two in a row.
        numeric_token = ((insert_space + street_num_1_to_3) | (insert_space + street_num_4_plus)).optimize()

        non_numeric_token = (
            ordinal_processor
            | english_word_processor
            | (insert_space + joined_num)
            | (insert_space + slashed_num_letter)
            | letter_processor
            | pynini.accep(NEMO_SPACE)
            | comma_processor
            | (insert_space + standalone_hyphen)
            | other_word_processor
        ).optimize()

        # Forbids two numeric tokens back-to-back, so a digit run is always one maximal match.
        full_string_processor = (
            pynini.closure(non_numeric_token | (numeric_token + pynini.closure(non_numeric_token, 1)))
            + pynini.closure(numeric_token, 0, 1)
        ).optimize()

        # Window-based context matching around address keywords for robust detection
        address_keywords_en = pynini.project(en_to_hi_map, "input")
        state_city_keywords = pynini.union(
            pynini.project(states_graph, "input"),
            pynini.project(cities_graph, "input"),
        ).optimize()
        address_keywords = pynini.union(address_keywords_hi, address_keywords_en, state_city_keywords)

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

        address_keywords_hi = pynini.string_file(get_abs_path("data/address/context.tsv"))
        states_graph = pynini.string_file(get_abs_path("data/address/states.tsv"))
        cities_graph = pynini.string_file(get_abs_path("data/address/cities.tsv"))
        en_to_hi_map = pynini.string_file(get_abs_path("data/address/en_to_hi_mapping.tsv"))

        address_graph = self.get_address_graph(
            cardinal, ordinal, input_case, address_keywords_hi, states_graph, cities_graph, en_to_hi_map
        )
        structured_address_graph = self.get_structured_address_graph(states_graph, cities_graph)

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
