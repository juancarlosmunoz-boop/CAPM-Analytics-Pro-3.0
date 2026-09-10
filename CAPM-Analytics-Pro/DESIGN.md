# UX/UI design decisions

## Product direction
CAPM Analytics Pro is presented as a financial research workspace rather than a calculator. The primary task is: configure a ticker, run the evidence chain, inspect risk, then export a documented CAPM case.

## Visual system
- Navy is the main structural color for institutional/professional hierarchy.
- Red is used sparingly for emphasis and risk/attention.
- Neutral surfaces create strong visual grouping without excessive decoration.
- 8px-based spacing rhythm is approximated through compact paddings and consistent vertical gaps.
- Cards have subtle borders/shadows; hover elevation provides restrained microinteraction.

## Information architecture
1. Analysis setup — controls the case.
2. Overview — price/return context and CAPM decomposition.
3. Beta Lab — regression, rolling beta, triangulation and quality diagnostics.
4. CAPM Engine — formula, Advanced CAPM evidence, SML and scenarios.
5. Risk & Report — lineage, country context, quality and downloads.

## Accessibility
Streamlit-native controls provide labels, keyboard access and focus states. Custom typography maintains high contrast. Color is not the only carrier of meaning: labels, text and table values accompany visual cues.

## Responsive behaviour
The dashboard uses Streamlit's responsive columns plus CSS breakpoints to reduce hero sizing, card density and metadata on smaller screens. The sidebar remains the main setup surface so the analytical content stays uncluttered.

## Institutional identity
The header references the Universidad del Cauca official symbol. The University's official symbols page and 2024 visual identity manual are the authority for the institution's identity usage.
