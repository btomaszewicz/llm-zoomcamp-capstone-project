# NOTES

Where do I add AI use?

Say what the scope is and what it cannot do
Chart of Architecture? description of "Request flow"? E.g. "3. The query router identifies conversation, knowledge, pricing, schedule, or availability intent."

STACK
Language	Python 3.12+
Web interface	Streamlit
API	Flask???????

Project structure

Data source

Something like "Evaluation criteria and rubric evidence: This table maps the course rubric directly to repository evidence so reviewers do not need to infer where each requirement is implemented."




# SUGGESTED OUTLINE

# Clinical Synopsis

Clinical Synopsis is a RAG application that transforms structured synthetic patient records—such as tables and spreadsheets of conditions, medications, procedures, diagnostic reports, and oncology events—into plain-language clinical summaries and answers.

Users can ask questions in natural language, review source-grounded responses, and inspect the underlying retrieved patient records for verification.

This project uses synthetic data and is a technical demonstration only. It is not intended for clinical decision-making.



## Problem

Oncology patient records are clinically rich but difficult to review as a coherent narrative. Relevant information is distributed across structured electronic health record data such as encounters, conditions, medications, procedures, diagnostic reports, laboratory results, and oncology-related events. A clinician answering a specific question—such as *What treatment did the patient receive?* or *Is there evidence of progression?*—may need to connect information across multiple document types and time periods.

This project uses synthetic breast-cancer patient records from the mCODE STU2 lifetime/longitudinal dataset. mCODE is an HL7 FHIR-based standard that defines structured data elements for oncology EHRs. The records include both cancer-related data and broader non-cancer medical history.

The challenge is therefore not only retrieving individual records. It is producing a concise, clinically useful synthesis from mixed structured sources while keeping the answer traceable to the evidence used. The application should retrieve the appropriate context, distinguish relevant from irrelevant history, and make the underlying source records available for review.

## Solution

Clinical Synopsis converts selected structured synthetic patient records into a searchable patient corpus with document metadata. Users select a synthetic patient and ask a clinical question in natural language.

The application then:

1. Routes the question to a supported question type, such as patient overview, conditions, medications, or oncology timeline.
2. Retrieves relevant patient-specific evidence using hybrid, semantic, or lexical search.
3. Builds question-type-aware context from the retrieved records.
4. Generates a concise clinical answer from that context.
5. Displays the available underlying health-record sources so the answer can be reviewed against the retrieved evidence.
6. Collects optional answer feedback and the monitoring dashboard diplays quality, cost, token, routing and judge evaluation metrics.

The project evaluates both retrieval and generation. Retrieval is evaluated against manually defined ground truth. The generated answers are evaluated by reference-based LLM judge for relevance and source-grounding. 

Human review remains necessary because automated evaluation alone cannot establish clinical correctness. The Monitoring page compares clinician feedback with the LLM judge’s relevance and groundedness assessments to identify agreement and disagreement between the two.

## Demo

Screenshot or short GIF of the Streamlit application.


## Data and knowledge base

The source dataset is the original synthetic mCODE/Synthea data; the knowledge base is the processed, chunked, metadata-enriched, and indexed patient corpus that the RAG system searches.

The project uses a selected sample of 50 synthetic breast-cancer patient records from the mCODE STU2 lifetime/longitudinal dataset. The original FHIR resources are transformed into derived patient-level Markdown and CSV records representing encounters, conditions, medications, procedures, diagnostic reports, and oncology-related timelines.

The derived records are split into chunks and stored with metadata including patient ID, document type, section heading, date range, and oncology flag. The same chunk metadata supports both a MinSearch lexical index and a local sentence-transformer vector index for semantic retrieval.



### MY INTERFACE FLOW / Summary of the demo
Clinical Synopsis
Generate clinical summaries with supporting evidence from available patient records.

Ask a clinical question
[ Text area ]

Question type
[ Router suggestion with optional override ]

[ Generate synopsis ]



**Request flow**
1. The user asks a question in the app.
2. The router chooses a question type or asks for clarification.
3. `rag_service` picks the corresponding prompt mode and retrieval scope from config.py.
4. retrieval.py fetches relevant chunks from the lexical, semantic, or hybrid index.
5. `build_context` formats those chunks into a prompt context.
6. The LLM generates the answer.
7. The judge evaluates grounding and relevance against the retrieved context.


## Features

- Synthetic patient selection
- Hybrid, semantic, and lexical retrieval
- Question routing
- Source-record previews
- Answer feedback collection
- Monitoring dashboard
- Retrieval and LLM-judge evaluation notebook

## Architecture

![Clinical Synopsis architecture](images/clinical-synopsis-architecture.png)

## Quick start

Exact `uv` commands to install and run the app.

## Evaluation

Explain:
- Gold chunk sets for retrieval evaluation
- Retrieval metrics
- Routed versus generic baseline
- LLM judge for relevance and groundedness
- Limits of automated judging

## Repository structure

Brief explanation of the main folders.

## Data and limitations

State that the project uses synthetic patient data and is a demonstration, not a clinical decision-support tool.

## Tech stack

Python, Streamlit, OpenAI API, Chroma/your vector store, Pandas, SQLite, uv.

## License

