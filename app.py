import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import os
from langfuse import Langfuse
langfuse = Langfuse()
from langfuse.decorators import observe
from langfuse.openai import OpenAI

from pycaret.regression import load_model


load_dotenv()
llm_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

###
# Loded model

loaded_model = load_model('model/best_gbr_model_halfmarathon')

st.set_page_config(
    page_title="Estymacja biegu",  # Ustaw tytuł strony
    page_icon="🏃🏻‍➡️",     # Możesz użyć emoji jako favicon
    # page_icon="ścieżka/do/ikonki.png",  # Lub podać ścieżkę do pliku z favicon
)

###
# TITLE
st.title('Aplikacja do estymacji czasu pół maratonu 🏃🏻‍➡️⏱️')

###
# AI Model function
@observe()
def get_chatbot_reply(user_prompt):
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
                    Jesteś pomocnikiem, któremu użytkownik płeć, wiek, oraz czas tempo na 5km.
                    Jeżeli w mojej wiadomości będzie informacja że jestem mężczyzną wypisz M, jeżeli
                    kobietą wypisz K.
                    Jeżeli ktoś użyje niestandardowej opisu płci wyłuskaj informacje jaka to płeć.
                    Następnie po przecinku, analogicznie do wieku dasz kategorię wiekową, bazujący czy M czy K skalowaną co 10 lat, czyli:
                    od M20 do M80 lub od K20 do K80.
                    Pamiętaj że użytkownik, może nie być dokładny i zamiast kropki dla tempo wpisać przecinek, potraktuj przecinek jako kropkę.
                    Jeżli kotś poda wiek 18 lub 19 lat potraktuj jak kategorię wiekową 20.
                    Jeżeli ktoś poda wiek poniżej 18 lat odpowiedz że model nie odwzoruje dobrze wyniku. 
                    Na koniec zwróć wynik w formacie:
                    {'Płeć': '...', 'Kategoria wiekowa': '...', '5 km Tempo': '...'}
                """
            },
            {"role": "user", "content": user_prompt}
        ]
    )

    response_content = response.choices[0].message.content.strip()

    # Debug the output from the model
    print("Odpowiedź z modelu:", response_content)

    if "nie podałeś wszystkich wymaganych danych" in response_content:
        raise ValueError("Nie podałeś wszystkich wymaganych danych.")
    if "podano niepoprawny format danych" in response_content:
        raise ValueError("Podano niepoprawny format danych.")

    try:
        input_data = eval(response_content)  # Cept eval if you are sure about the answer
    except (SyntaxError, NameError) as e:
        raise ValueError(f"Niepoprawny format odpowiedzi: {response_content}")

    required_keys = ['Płeć', 'Kategoria wiekowa', '5 km Tempo']
    for key in required_keys:
        if key not in input_data:
            raise ValueError(f"Brak klucza '{key}' w danych wejściowych.")

    return input_data

###
# Convert secunds to hh/hh/ss
def seconds_to_hhmmss(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

# Interface streamlit
with st.expander("📖 Instrukcja (kliknij, aby rozwinąć)"):
    st.write("""
        Wpisz kolejno swoją płeć, wiek, oraz Tempo na 5 km.
        Aplikacja wyestymuje twój przybliżony czas, jaki będzie potrzebny, 
        aby ukończyć półmaraton.
        Oceń ostateczny wynik.
             
        Estymacja wyników jest na bazie danych z półmaratonu wrocławskiego z lat 2023 i 2024
    """)

if 'user_input' not in st.session_state:
    st.session_state['user_input'] = []

for message in st.session_state['user_input']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("O co chcesz spytać?")
if prompt:
    user_message = {"role": "user", "content": prompt}
    with st.chat_message("human"):
        st.markdown(user_message["content"])

    st.session_state['user_input'].append(user_message)

    with st.chat_message("assistant"):
        try:
            input_data = get_chatbot_reply(prompt)
            predicted_seconds = loaded_model.predict(pd.DataFrame([input_data]))[0]
            predicted_time = seconds_to_hhmmss(predicted_seconds)

            # Add result to session_state
            st.session_state['user_input'].append({"role": "assistant", "content": f"Szacowany czas półmaratonu: {predicted_time}"})
            st.markdown(f"Szacowany czas półmaratonu: {predicted_time}")

        except ValueError as e:
            # Error handling when input is invalid
            st.markdown(str(e))