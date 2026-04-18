# MovieRecApp — Clean Showcase Build

A local movie recommendation app built from the Advanced Topics in Artificial Intelligence chatbot project.

It allows the user to look up facts about a movie - who directed it, which year did the release...

It also provides movie recommendations according to the movies, genre, director... a user likes.

## Menu

- [1. System overview](#1-system-overview)
  - [1.1 Extraction pipeline](#11-extraction-pipeline)
  - [1.2 Recommendation pipeline](#12-recommendation-pipeline)
  - [1.3 Factual logic](#13-factual-logic)
- [2. Install and run](#2-install-and-run)
  - [2.1 Prerequisites](#21-prerequisites)
  - [2.2 Create and activate a Python environment](#22-create-and-activate-a-python-environment)
  - [2.3 Install requirements](#23-install-requirements)
  - [2.4 Install the spaCy English model](#24-install-the-spacy-english-model)
  - [2.5 Run the web app](#25-run-the-web-app)
  - [2.6 [Optional] Run the direct command-line interface](#26-run-the-direct-cli)

---

## 1. System overview

This app has three main runtime layers:

1. **message routing**
   - `ApproachIdentifier` decides whether a message looks like a recommendation request or a factual question.
2. **local extraction + resolution**
   - movie titles and entities are extracted from the user message and mapped into the app’s local dictionaries.
3. **local retrieval / ranking**
   - the app scores candidates from prebuilt artifacts and returns either recommendations or factual answers.

The runtime entry points are:

- **web app**: `webapp/app.py`
- **Command-Line Interface**: `src/main.py`
- **main agent**: `src/modules/agent.py`

The agent uses these runtime resources:

- `dataset/runtime_artifacts/`
- `dataset/ratings/`
- `dataset/titles_to_qid.json`

### 1.1 Extraction pipeline

Recommendation extraction is handled by `Extractor.extract_rec_mulm(...)` in `src/modules/extractors.py`.

The flow is:

1. **extract movie titles first**
   - the app calls `debug_extract_titles(...)` from `src/modules/title_extraction_latest.py`
   - extracted movie titles are removed from the message with placeholders before entity extraction continues
   - short titles that are risky or ambiguous can be deferred and resolved later

2. **extract release years from the remaining text**
   - remaining text is scanned with a year regex
   - years such as `2018` are turned into synthetic release-year entities and later mapped to `P577` release dates rather than “set in period”

3. **collect signal token sequences**
   - spaCy POS tagging is used to keep mainly `NOUN`, `PROPN`, and `ADJ` tokens
   - aliases such as `sci-fi`, `scifi`, and `sci fi` are normalized to `science fiction`

4. **generate candidate entity spans**
   - the extractor builds candidate subspans up to length 5
   - confounding words such as `movie`, `movies`, `film`, `films`, `recommend`, `show`, and similar carrier words are stripped or filtered out

5. **suppress subspans under longer exact matches**
   - if a longer span already has an exact dictionary match, shorter subspans inside it are suppressed
   - example: if `jake gyllenhaal` matches exactly, the inner span `jake` is not kept as a separate exact entity hit

6. **resolve entities**
   - exact entity-dictionary matches are checked first
   - then synonym-based matches are checked through `EntitySynonymMatcher`
   - genre/tag matching can expand through the prebuilt genre-tag bag index

7. **rescue deferred short titles**
   - after entity extraction, deferred short movie titles are revisited
   - if they still look safe, they are rescued as titles; otherwise they are returned as clarification candidates

The extraction result is a dictionary containing fields such as:

- `titles`
- `entities`
- `left_over`
- `clarify_titles`
- `rescued_short_titles`

### 1.2 Recommendation pipeline

Recommendation handling lives mainly in `Agent._run_recommendation_logic(...)` and `ContentRecommender` in `src/modules/recommender/recommender.py`.

The flow is:

1. **extract titles and entities**
   - the agent calls `extract_rec_mulm(...)`
   - titles are resolved to movie QIDs through `TitleResolver`
   - entity labels are mapped to internal IDs through the local dictionaries

2. **build a preference profile**
   - `ContentRecommender.build_preference_profile(...)` builds a bag-of-attributes profile from:
     - liked movie titles
     - explicitly mentioned entities
   - for liked movie titles, the profile is based on `movie_attrs.pkl`

3. **weight repeated features across liked movies**
   - if the same attribute/value appears in multiple liked movies, it is boosted
   - if an attribute/value appears in only one liked movie, it is penalized relative to the total number of liked movies
   - cast-member weighting can use `dataset/top-500-celeb.json` when that file is present

4. **resolve explicit entity constraints**
   - explicit entities are matched back into raw attribute values through `attr_index.pkl`
   - genre/tag constraints can expand through the prebuilt phrase/token lookup artifact
   - release-year entities are matched against release-date values and compared by year

5. **try checkbox-style constrained recommendation first**
   - if entity constraints are present, `recommend_with_constraint_groups(...)` builds one candidate set per entity
   - it first tries a full intersection
   - if full intersection fails, it falls back to maximal groups of satisfied constraints
   - for entity-only requests, this path includes popularity-tier-aware randomization so repeated prompts are not forced to return the exact same movie list every time

6. **fallback to profile scoring + KNN refinement**
   - if the grouped constraint path does not fully handle the request, the app scores movies by profile overlap
   - it then keeps the top content-based candidates
   - after that, it tries a KNN refinement step over those candidates using the rating matrix
   - if KNN produces nothing, it falls back to the content-ranked list
   
7. **format the final reply**
   - titles are mapped back to human-readable movie names
   - for multi-movie inputs, the reply may also include a short strongest-features report based on high-scoring shared attributes

In short, the recommendation system is a **hybrid local recommender**:

- title extraction
- entity extraction
- attribute-profile scoring
- checkbox/group filtering for entity constraints
- KNN refinement

### 1.3 Factual logic

Factual logic is handled by `ApproachIdentifier`, `IntentIdentifier`, and the factual lookup methods in `Agent`.

The flow is:

1. **detect factual intent**
   - `ApproachIdentifier` checks whether the message looks factual instead of recommendational
   - examples include questions about directors, writers, release date/year, cast, genre, country, language, awards, composer, cinematographer, editor, filming location, box office, and similar movie metadata

2. **map the message to an intent**
   - `IntentIdentifier.question_to_intent(...)` uses regex patterns from `PatternLibrary`
   - it maps a question to an `IntentSpec`, which includes:
     - the relation IRI
     - the internal intent name
     - whether the answer is expected to be a literal or an entity

3. **resolve the entity label locally**
   - the extracted entity label is resolved with the local entity linker
   - if resolution fails, the app returns a local-dataset miss instead of calling an external service

4. **answer from local runtime artifacts**
   - `_lookup_factual(...)` reads from:
     - `movie_attrs.pkl`
     - `attr_index.pkl`
   - for inverse lookups such as “movies by director”, it uses `attr_index`
   - for forward lookups such as “who directed X?”, it uses `movie_attrs`
   - release years are derived from `P577` release-date values

Important runtime property:

- **the app does not query the RDF graph at runtime**
- all factual answers come from the local prebuilt artifacts and local dictionaries
- This design allows the app to be uploaded in its entirety onto a remote repository and be shared. This was not previously possible due to the massive size of the RDF graph. The reduction of the overall size of the app also allows future possibility of deployment and hosting the app completely online.
---

## 2. Install and run

### 2.1 Prerequisites

You need:

- Python 3.x


### 2.2 Create and activate a Python environment

#### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Windows (cmd)

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2.3 Install requirements

```bash
pip install -r requirements-webapp.txt
```

### 2.4 Install the spaCy English model

The extractor uses spaCy POS tagging, so you also need the English model once:

```bash
python -m spacy download en_core_web_sm
```

If your shell uses `python3`, use:

```bash
python3 -m spacy download en_core_web_sm
```

### 2.5 Run the web app

From the project root:

```bash
python run_webapp.py
```

Then open:

- `http://127.0.0.1:8000/`

What this starts:

- Flask app: `webapp/app.py`
- host: `0.0.0.0`
- port: `8000`
- debug mode: on

Useful live route:

- health check: `http://127.0.0.1:8000/api/health`

### 2.6 Run the direct CLI

If you want to test the same agent without the browser UI:

```bash
python src/main.py
```

This starts a simple terminal prompt where you can type movie questions directly.

Type one of these to exit:

- `quit`
- `exit`
- blank line

---
