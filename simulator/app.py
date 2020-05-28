import streamlit as st

import pages.seir
import pages.about

PAGES = {
    "Modelo epidemiológico": pages.seir,
    #"Sobre o projeto": pages.about,
}


def main():
    logo_html = pages.utils.texts.insert_logos()
    st.markdown(
        logo_html, unsafe_allow_html=True,
    )
    st.markdown(pages.utils.texts.INTRODUCTION)

   
    #goto = st.sidebar.radio("Ir para", list(PAGES.keys()))
    PAGES["Modelo epidemiológico"].write()

if __name__=="__main__":
    main()
