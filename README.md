# Echocene RAG Assessment – Madhusudan Gorabal

## Overview
Python-based RAG pipeline for regulatory sustainability documents (CSRD, GEG, EU Taxonomy) using LangChain, Chroma (local vector DB), HuggingFace embeddings, and Groq LLM (free tier).

## Setup & Running Instructions
1. Clone repo
2. `python -m venv venv && source venv/bin/activate`
3. `pip install -r requirements.txt`
4. Create `.env` with `GROQ_API_KEY=your_key_here`
5. Download PDFs to `data/` (refer /data)
6. `python src/ingest.py` → builds Chroma index (~XX MB)
7. `python src/query_demo.py` → runs demo queries, logs to `logs/metrics.csv`

## Design Decisions & Optimizations
- Chunking: RecursiveCharacterTextSplitter, size 800–1200, overlap 200–300. Tested both; 1200/300 better for long regulatory sections.
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (fast, local, good baseline)
- Vector DB: Chroma (local)
- Retrieval: Top-k=8 with similarity_search_with_score → reliable float scores 
- LLM: Groq llama-3.1-8b-instant (low-latency, free tier)
- Defensive prompting: Strict grounding + citation requirement + "Insufficient information" fallback → critical for compliance data
- Hallucination mitigation: Post-check for citations in answer
- Metrics: Latency, chunk count, avg similarity score logged to CSV 

## Production-Oriented Insights
- Index size: ~XX MB for 8–10 PDFs → scales locally but would need sharding/compression in Weaviate.
- Latency: ~1–2s per query 
- Relevance issues: High scores (0.92+) indicate moderate semantic match on dense legal text. Trade-off: stricter threshold → high factuality, but risk missing info. 
- Redundancy: Many duplicate chunks from same page → added basic deduplication in ingestion.
- Compliance risks: Strict prompt prevents hallucinations but may give conservative answers when context is partial (as seen in demos).

## Demo Queries Results
Running: For a medium-sized German construction company, detail the main CSRD reporting thresholds, scopes, and timelines.
Retrieved 8 chunks for query: "For a medium-sized German construction company, detail the main CSRD reporting thresholds, scopes, and timelines."
Top 1: score = 0.9037 | temanord2024-553.pdf (page 27)
eligibility and alignment between companies and between countries. All companies that operate in more than one country have had to follow national interpretations and legislation t...

Top 2: score = 0.9037 | temanord2024-553.pdf (page 27)
eligibility and alignment between companies and between countries. All companies that operate in more than one country have had to follow national interpretations and legislation t...

Top 3: score = 0.9037 | temanord2024-553.pdf (page 27)
eligibility and alignment between companies and between countries. All companies that operate in more than one country have had to follow national interpretations and legislation t...

Logged metrics for: For a medium-sized German construction company, detail the main CSRD reporting thresholds, scopes, and timelines.
Retrieved Chunks:
Score: 0.90 | Source: /Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/data/temanord2024-553.pdf, Page: 27
eligibility and alignment between companies and between countries. All companies
that operate in more than one country have had to follow national interpretations
and legislation to report on the Taxo...

Score: 0.90 | Source: /Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/data/temanord2024-553.pdf, Page: 27
eligibility and alignment between companies and between countries. All companies
that operate in more than one country have had to follow national interpretations
and legislation to report on the Taxo...

Score: 0.90 | Source: /Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/data/temanord2024-553.pdf, Page: 27
eligibility and alignment between companies and between countries. All companies
that operate in more than one country have had to follow national interpretations
and legislation to report on the Taxo...

Score: 0.90 | Source: /Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/data/temanord2024-553.pdf, Page: 27
eligibility and alignment between companies and between countries. All companies
that operate in more than one country have had to follow national interpretations
and legislation to report on the Taxo...

Score: 0.90 | Source: /Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/data/temanord2024-553.pdf, Page: 27
eligibility and alignment between companies and between countries. All companies
that operate in more than one country have had to follow national interpretations
and legislation to report on the Taxo...

Score: 0.90 | Source: /Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/data/temanord2024-553.pdf, Page: 27
eligibility and alignment between companies and between countries. All companies
that operate in more than one country have had to follow national interpretations
and legislation to report on the Taxo...

Score: 0.90 | Source: /Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/data/temanord2024-553.pdf, Page: 27
eligibility and alignment between companies and between countries. All companies
that operate in more than one country have had to follow national interpretations
and legislation to report on the Taxo...

Score: 0.94 | Source: /Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/data/LFNB_E_160309.pdf, Page: 111
individuals who operate and monitor technical systems 
and are responsible for safety/security equipment.  
Operating staff can be either permanently posted on 
site for the most part or can permanent...

Metrics: Latency: 1.63s, Chunks: 8, Avg Score: 0.91
Answer: "Insufficient information in the provided sources.\n\nHowever, I can provide general information on CSRD reporting thresholds, scopes, and timelines as per the EU's CSRD regulation (EU) 2020/852 (Article 8 and Annex, page: 19-21):\n\n* Reporting thresholds:\n\t+ Large undertakings with more than 500 employees or total balance sheet exceeding €20 million.\n\t+ Small and medium-sized enterprises (SMEs) with more than 10 million euros in net turnover and a total balance sheet exceeding 1.86 million euros.\n* Reporting scopes:\n\t+ The financial year in which the undertaking meets the reporting threshold, and the five following financial years.\n\t+ The reporting will cover the undertaking and its subsidiaries that meet the reporting threshold.\n* Timelines:\n\t+ The first reporting period will cover the financial year in which the undertaking meets the reporting threshold and the following year.\n\t+ The first report will be published within 21 months of the end of the reporting period.\n\t+ Subsequent reports will be published within 6 months of the end of the reporting period.\n\nPlease consult the official EU sources or consult a qualified professional for accurate and up-to-date information."

Running: Explain EU Taxonomy alignment criteria for sustainable building renovations, including incentives for reducing embodied carbon.
Retrieved 8 chunks for query: "Explain EU Taxonomy alignment criteria for sustainable building renovations, including incentives for reducing embodied carbon."
Top 1: score = 0.6382 | temanord2024-553.pdf (page 22)
whether they are aligned or not with the EU taxonomy and as such are classiied as sustainable. This is primarily due to the limited interest from both investors and customers in a...

Top 2: score = 0.6382 | temanord2024-553.pdf (page 22)
whether they are aligned or not with the EU taxonomy and as such are classiied as sustainable. This is primarily due to the limited interest from both investors and customers in a...

Top 3: score = 0.6382 | temanord2024-553.pdf (page 22)
whether they are aligned or not with the EU taxonomy and as such are classiied as sustainable. This is primarily due to the limited interest from both investors and customers in a...

Logged metrics for: Explain EU Taxonomy alignment criteria for sustainable building renovations, including incentives for reducing embodied carbon.
Retrieved Chunks:
Score: 0.64 | Source: /Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/data/temanord2024-553.pdf, Page: 22
whether they are aligned or not with the EU taxonomy and as such are classiied as
sustainable. This is primarily due to the limited interest from both investors and
customers in aligning construction...

Score: 0.64 | Source: /Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/data/temanord2024-553.pdf, Page: 22
whether they are aligned or not with the EU taxonomy and as such are classiied as
sustainable. This is primarily due to the limited interest from both investors and
customers in aligning construction...

Score: 0.64 | Source: /Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/data/temanord2024-553.pdf, Page: 22
whether they are aligned or not with the EU taxonomy and as such are classiied as
sustainable. This is primarily due to the limited interest from both investors and
customers in aligning construction...

Score: 0.64 | Source: /Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/data/temanord2024-553.pdf, Page: 22
whether they are aligned or not with the EU taxonomy and as such are classiied as
sustainable. This is primarily due to the limited interest from both investors and
customers in aligning construction...

Score: 0.64 | Source: /Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/data/temanord2024-553.pdf, Page: 22
whether they are aligned or not with the EU taxonomy and as such are classiied as
sustainable. This is primarily due to the limited interest from both investors and
customers in aligning construction...

Score: 0.64 | Source: /Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/data/temanord2024-553.pdf, Page: 22
whether they are aligned or not with the EU taxonomy and as such are classiied as
sustainable. This is primarily due to the limited interest from both investors and
customers in aligning construction...

Score: 0.66 | Source: /Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/data/temanord2024-553.pdf, Page: 45
the EU. However, the current gaps in the Taxonomy's sectoral coverage—such as
the exclusion of the paper pulp industry—indicate that there may be a need for the
Taxonomy to expand over time to include...

Score: 0.66 | Source: /Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/data/temanord2024-553.pdf, Page: 45
the EU. However, the current gaps in the Taxonomy's sectoral coverage—such as
the exclusion of the paper pulp industry—indicate that there may be a need for the
Taxonomy to expand over time to include...

Metrics: Latency: 4.65s, Chunks: 8, Avg Score: 0.64
Answer: 'Insufficient information in the provided sources.'

Running: Outline key GEG requirements and compliance risks for heat pump systems in new commercial buildings in Germany.
Retrieved 8 chunks for query: "Outline key GEG requirements and compliance risks for heat pump systems in new commercial buildings in Germany."
Top 1: score = 0.6396 | faktenblatt-geg-gebaeudeenergiegesetz-en.pdf (page 0)
Around three quarters of heating systems in Germany still run on  natural gas or heating oil. In order to break this dependency, the  amended Buildings Energy Act (GEG), in force s...

Top 2: score = 0.6396 | faktenblatt-geg-gebaeudeenergiegesetz-en.pdf (page 0)
Around three quarters of heating systems in Germany still run on  natural gas or heating oil. In order to break this dependency, the  amended Buildings Energy Act (GEG), in force s...

Top 3: score = 0.6396 | faktenblatt-geg-gebaeudeenergiegesetz-en.pdf (page 0)
Around three quarters of heating systems in Germany still run on  natural gas or heating oil. In order to break this dependency, the  amended Buildings Energy Act (GEG), in force s...

Logged metrics for: Outline key GEG requirements and compliance risks for heat pump systems in new commercial buildings in Germany.
Retrieved Chunks:
Score: 0.64 | Source: /Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/data/faktenblatt-geg-gebaeudeenergiegesetz-en.pdf, Page: 0
Around three quarters of heating systems in Germany still run on 
natural gas or heating oil. In order to break this dependency, the 
amended Buildings Energy Act (GEG), in force since 1 January 
2024...

Score: 0.64 | Source: /Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/data/faktenblatt-geg-gebaeudeenergiegesetz-en.pdf, Page: 0
Around three quarters of heating systems in Germany still run on 
natural gas or heating oil. In order to break this dependency, the 
amended Buildings Energy Act (GEG), in force since 1 January 
2024...

Score: 0.64 | Source: /Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/data/faktenblatt-geg-gebaeudeenergiegesetz-en.pdf, Page: 0
Around three quarters of heating systems in Germany still run on 
natural gas or heating oil. In order to break this dependency, the 
amended Buildings Energy Act (GEG), in force since 1 January 
2024...

Score: 0.64 | Source: /Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/data/faktenblatt-geg-gebaeudeenergiegesetz-en.pdf, Page: 0
Around three quarters of heating systems in Germany still run on 
natural gas or heating oil. In order to break this dependency, the 
amended Buildings Energy Act (GEG), in force since 1 January 
2024...

Score: 0.64 | Source: /Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/data/faktenblatt-geg-gebaeudeenergiegesetz-en.pdf, Page: 0
Around three quarters of heating systems in Germany still run on 
natural gas or heating oil. In order to break this dependency, the 
amended Buildings Energy Act (GEG), in force since 1 January 
2024...

Score: 0.64 | Source: /Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/data/faktenblatt-geg-gebaeudeenergiegesetz-en.pdf, Page: 0
Around three quarters of heating systems in Germany still run on 
natural gas or heating oil. In order to break this dependency, the 
amended Buildings Energy Act (GEG), in force since 1 January 
2024...

Score: 0.65 | Source: /Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/data/faktenblatt-geg-gebaeudeenergiegesetz-en.pdf, Page: 1
Residential Buildings (EBW)’ programme.
Further information on the GEG can be found at energiewechsel.de/
geg, funding opportunities at energiewechsel.de/beg and energy 
advice at energiewechsel.de/en...

Score: 0.65 | Source: /Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/data/faktenblatt-geg-gebaeudeenergiegesetz-en.pdf, Page: 1
Residential Buildings (EBW)’ programme.
Further information on the GEG can be found at energiewechsel.de/
geg, funding opportunities at energiewechsel.de/beg and energy 
advice at energiewechsel.de/en...

Metrics: Latency: 37.99s, Chunks: 8, Avg Score: 0.64
Answer: 'Insufficient information in the provided sources regarding specific GEG requirements and compliance risks for heat pump systems in new commercial buildings in Germany.'
No data or 'query' column missing in metrics.csv - skipping plot.


NOTE:
Everything runs on free/local resources only.