# CAPM Analytics Pro — V2 UX/UI

A Streamlit equity-risk research workspace centered on the CAPM equation:

`E(Ri) = Rf + beta * [E(Rm) - Rf]`

## UX/UI direction
- Institutional-inspired navy/red visual system.
- Desktop-first dashboard with responsive CSS for tablet/mobile widths.
- Clear setup form in the sidebar; the main area becomes an evidence workspace after execution.
- Reusable metric cards, source chips, callouts, data tables and interactive Plotly charts.
- Accessible labels, visible feedback, tooltips/help, keyboard-friendly Streamlit controls and high-contrast text.
- Microinteraction via hover elevation on metric cards and non-blocking toast feedback.

## Analysis flow
1. Ticker + benchmark from Yahoo Finance.
2. Synchronized returns and historical OLS beta.
3. Damodaran country ERP/CRP and global industry beta.
4. FRED U.S. Treasury 10Y risk-free rate (or manual fallback/input).
5. CAPM required return.
6. Rolling beta, regression, SML, sensitivity and scenarios.
7. Beta Quality Score and data lineage.
8. PDF + CSV exports.

## Deployment
Upload the project contents to GitHub and deploy `app.py` on Streamlit Community Cloud. Dependencies are declared in `requirements.txt`.

The app uses live external sources at runtime. Local unit tests validate the core logic and adapter contracts without needing outbound Internet.

## Institutional identity
The UI references the Universidad del Cauca symbol. Official institutional symbols and identity guidance are published by Universidad del Cauca: https://www.unicauca.edu.co/la-universidad/simbolos/
