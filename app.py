import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Optional, Dict, Set
import itertools

# Get OpenAI API key from environment variable or Streamlit secrets
if 'OPENAI_API_KEY' not in st.session_state:
    st.session_state['OPENAI_API_KEY'] = st.secrets.get('OPENAI_API_KEY', '')

# Function schema for structured output
WORD_SCHEMA = {
    "name": "generate_word",
    "description": "Generate a new word for an abstract emotion using specified languages",
    "parameters": {
        "type": "object",
        "properties": {
            "word": {
                "type": "string",
                "description": "The generated word"
            },
            "pronunciation": {
                "type": "string",
                "description": "Pronunciation guide in IPA format"
            },
            "definition": {
                "type": "string",
                "description": "One-sentence poetic definition"
            },
            "etymology": {
                "type": "string",
                "description": "Breakdown of word roots and origins"
            },
            "examples": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Two example sentences using the word",
                "minItems": 2,
                "maxItems": 2
            }
        },
        "required": ["word", "pronunciation", "definition", "etymology", "examples"]
    }
}

# Language categories and complementary pairs
LANGUAGE_OPTIONS = {
    "Ancient Languages": {
        "Latin": "Systematic word formation and scientific precision",
        "Ancient Greek": "Complex abstract concepts and philosophical depth",
        "Sanskrit": "Rich compound system and spiritual concepts"
    },
    "Modern Languages": {
        "German": "Excellent compound formation and precision",
        "Japanese": "Emotional nuance and harmonious sounds",
        "Arabic": "Poetic roots and abstract concepts",
        "Mandarin": "Concise forms and tonal expression"
    },
    "Historical Languages": {
        "Old English": "Anglo-Saxon strength and earthiness",
        "Old Norse": "Nature and mythological imagery",
        "Classical Persian": "Poetic beauty and emotional depth"
    },
    "Fictional Languages": {
        "Elvish (Quenya/Sindarin)": "Ethereal elegance and nature harmony",
        "Klingon": "Honor-focused and direct expression"
    }
}

# Define complementary language pairs with explanations
COMPLEMENTARY_PAIRS = {
    ("Latin", "Japanese"): "Precision meets emotion - great for technical feelings",
    ("Ancient Greek", "Old Norse"): "Philosophical depth with mythological power",
    ("Sanskrit", "Elvish"): "Spiritual depth meets natural harmony",
    ("German", "Arabic"): "Structural precision meets poetic flow",
    ("Old English", "Classical Persian"): "Earthiness meets mystical beauty",
    ("Mandarin", "Klingon"): "Subtle tones meet warrior directness",
    ("Latin", "German"): "Classical precision with modern compound power",
    ("Japanese", "Elvish"): "Emotional subtlety with ethereal beauty",
    ("Sanskrit", "Old Norse"): "Ancient wisdom meets primal force",
    ("Arabic", "Classical Persian"): "Rich poetic traditions combined",
    ("Ancient Greek", "Elvish"): "Philosophical concepts with mythical elegance",
    ("Old English", "Klingon"): "Germanic strength meets warrior culture"
}

SYSTEM_PROMPT = """You are a precise linguist creating new words for abstract emotions. Your goal is to combine roots that directly relate to the core meaning of the described emotion.

Key Rules:
1. CRITICAL: Use ONLY roots from the specified languages
2. Select roots whose meanings DIRECTLY relate to the key components of the emotion:
   - Identify the core concepts in the emotional description
   - Choose roots that specifically express those concepts
   - Avoid roots with tangential or metaphorical connections
3. Make it pronounceable in English
4. For single language selections, use ONLY that language's roots

Example for "the joy of shared discovery":
✓ GOOD: German 'Gemein' (shared) + Greek 'chara' (joy)
✗ BAD: German 'Freude' (joy) + Greek 'eleutheria' (freedom)
   [Freedom is not directly related to the concept of sharing or discovery]

Example for "the comfort of morning sunlight":
✓ GOOD: Japanese 'asa' (morning) + Persian 'noor' (light)
✗ BAD: Japanese 'kaze' (wind) + Persian 'roshani' (brightness)
   [Wind is not part of the core concept being described]

For each word creation:
1. First identify the key concepts in the emotion
2. Then find roots that DIRECTLY express those concepts
3. Only combine roots that clearly relate to the described feeling"""


def get_suggested_pairs(selected_lang: str) -> List[tuple]:
    """Return suggested language pairs for a selected language."""
    suggestions = []
    for pair, description in COMPLEMENTARY_PAIRS.items():
        if selected_lang in pair:
            other_lang = pair[1] if selected_lang == pair[0] else pair[0]
            suggestions.append((other_lang, description))
    return suggestions


def generate_word(user_input: str, languages: List[str]) -> Optional[dict]:
    try:
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            api_key=st.session_state['OPENAI_API_KEY']
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", """Create a word for: "{input}"
Using STRICTLY ONLY these languages: {languages}

Requirements:
- Use ONLY roots from the specified languages above - no exceptions
- Include 2 vivid usage examples
- Make etymology clear and precise, citing only from the selected languages
- Ensure NO other language influences are included""")
        ])

        chain = prompt | llm.bind(functions=[WORD_SCHEMA],
                                  function_call={"name": "generate_word"}) | JsonOutputFunctionsParser()

        response = chain.invoke({
            "input": user_input,
            "languages": ", ".join(languages)
        })

        return response

    except Exception as e:
        st.error(f"Error generating word: {str(e)}")
        return None


def main():
    st.title("🌌 Reverse Dictionary of Feelings")
    st.caption("Describe an abstract emotion and select root languages")

    if not st.session_state['OPENAI_API_KEY']:
        st.error("Please set the OPENAI_API_KEY in your environment or Streamlit secrets.")
        st.stop()

    # Language selection interface
    col1, col2 = st.columns([2, 1])

    with col1:
        # # Language categories display
        # for category, langs in LANGUAGE_OPTIONS.items():
        #     st.subheader(f"🔹 {category}")
        #     for lang, tooltip in langs.items():
        #         st.caption(f"• {lang}: {tooltip}")
        #
        # st.divider()

        # Language selector
        all_languages = [lang for langs in LANGUAGE_OPTIONS.values() for lang in langs.keys()]
        selected_langs = st.multiselect(
            "Choose root languages (2-3 recommended):",
            options=all_languages
        )

    # with col2:
    #     if len(selected_langs) == 1:
    #         st.subheader("✨ Suggested Pairs")
    #         suggestions = get_suggested_pairs(selected_langs[0])
    #         if suggestions:
    #             for lang, desc in suggestions:
    #                 st.info(f"**{selected_langs[0]} + {lang}**\n{desc}")
    #         else:
    #             st.info("No specific suggestions for this language, but feel free to experiment!")
    #
    #     elif len(selected_langs) == 0:
    #         st.info("Select a language to see suggested combinations!")

    # Display recommendation for number of languages
    if len(selected_langs) == 1:
        st.warning(
            "Using a single language is possible but combining 2-3 languages is recommended for more interesting results.")

    # Emotion input
    user_input = st.text_area(
        "Describe your feeling:",
        placeholder="e.g., 'The melancholy of abandoned amusement parks'",
        height=100
    )

    if st.button("Generate Word"):
        if not user_input.strip():
            st.error("Please describe an emotion!")
        elif not selected_langs:
            st.error("Please select at least one language!")
        else:
            with st.spinner("Weaving linguistic magic..."):
                result = generate_word(user_input, selected_langs)

                if result:
                    # Display results
                    st.markdown(f"## {result['word']} /{result['pronunciation']}/")
                    st.write(f"**{result['definition']}**")
                    st.caption(f"*Etymology*: {result['etymology']}")

                    if result['examples']:
                        st.divider()
                        st.write("**Examples in use:**")
                        for i, example in enumerate(result['examples'], 1):
                            st.write(f"{i}. {example}")

    # Example suggestions
    with st.expander("📝 Example Feelings to Try"):
        st.write("""
        ### Contemplative Moments
        - The peace of watching snow fall in complete silence *(German + Japanese)*
        - The strange comfort of being alone in a vast library *(Latin + Old English)*
        - The mysterious feeling of walking through morning mist *(Old Norse + Elvish)*

        ### Social Emotions
        - The joy of finding someone who shares your obscure interest *(Ancient Greek + German)*
        - The warmth of laughing at an old memory with a friend *(Classical Persian + Sanskrit)*
        - The unique bond formed through sharing a meal in silence *(Japanese + Arabic)*

        ### Nostalgic Sentiments
        - The bittersweet feeling of looking at old photographs *(Japanese + Classical Persian)*
        - The melancholy of revisiting a place from your childhood *(German + Old English)*
        - The strange nostalgia for a time you've never experienced *(Ancient Greek + Elvish)*

        ### Modern Life
        - The satisfaction of closing all browser tabs after finishing a project *(Latin + Klingon)*
        - The anxiety of hearing your phone buzz during a meeting *(German + Mandarin)*
        - The relief of finding your keys exactly where you left them *(Sanskrit + Old Norse)*

        ### Nature & Cosmos
        - The humbling feeling of stargazing on a clear night *(Ancient Greek + Elvish)*
        - The primal joy of standing in a summer rainstorm *(Old Norse + Classical Persian)*
        - The profound peace of watching leaves dance in the wind *(Sanskrit + Japanese)*

        ### Creative Moments
        - The flow state when lost in creating something *(Ancient Greek + German)*
        - The frustration of having the perfect word on the tip of your tongue *(Latin + Arabic)*
        - The satisfaction of finally solving a complex puzzle *(Sanskrit + Klingon)*
        """)


if __name__ == "__main__":
    main()