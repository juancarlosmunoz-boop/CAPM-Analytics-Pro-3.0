# Fix: React/Streamlit `removeChild` NotFoundError

The visible error:

`NotFoundError: Failed to execute 'removeChild' on 'Node'`

is a browser-side React/DOM error, not a Python exception. A common trigger is browser translation software (including Google Translate) rewriting text nodes while React is updating the page. The supplied screenshot shows Chrome's translation control, which makes this a strong diagnostic lead.

The V2.2 application also removes several patterns that make rerendering more fragile:

- no explicit `st.rerun()` after analysis;
- no `st.status` context followed by a rerun;
- no `st.tabs()` containing multiple dynamic Plotly/dataframe trees;
- no custom HTML image `onerror` DOM mutation;
- analysis is computed once and stored in `st.session_state`;
- navigation uses a simple Streamlit radio control;
- charts/dataframes are rendered from stable Streamlit containers.

This is a defensive application-side cleanup. If the same browser error remains, first disable page translation for the deployed Streamlit URL or test the app in an incognito window with extensions disabled. Streamlit itself is not the component named in the browser stack trace; the stack is from the React frontend.
