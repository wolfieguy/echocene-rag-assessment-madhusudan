# Echocene RAG Assessment – Madhusudan Gorabal

## Overview
Python-based RAG pipeline for regulatory sustainability documents (CSRD, GEG, EU Taxonomy) using LangChain, Chroma (local vector DB), HuggingFace embeddings, and Groq LLM (free tier).

## Setup & Running Instructions
1. Clone repo
2. `python -m venv venv && source venv/bin/activate`
3. `pip install -r requirements.txt`
4. Create `.env` with `GROQ_API_KEY=your_key_here`
5. Download PDFs to `data/` 
6. `python src/ingest.py` -> builds Chroma index (~446 MB)
7. `python src/query_demo.py` -> runs demo queries, logs to `logs/metrics.csv`

## Design Decisions & Optimizations
- Chunking: RecursiveCharacterTextSplitter (tested 800/200 - 1500/400) -> Larger chunks (1500) better for long regulatory sections.
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (fast, local, good baseline) -> Moderate semantic match on legal text (avg scores 0.6–0.9).
- Vector DB: Chroma (local)
- Retrieval: Top-k=8 with similarity_search_with_score -> reliable float scores 
- LLM: Groq llama-3.1-8b-instant (low-latency, free tier)
- Retrieval: Hybrid-> BM25 keyword + semantic 
- Defensive prompting: Strict grounding + citation requirement + "Insufficient information" fallback -> critical for compliance data
- Hallucination mitigation: Post-check for citations in answer
- Metrics: Latency, chunk count, avg similarity score logged to CSV 

## Production-Oriented Insights
- Index size: ~446 MB for 8 PDFs after dedup and larger chunks. Local Chroma works well for prototype. 
- Latency: ~1–2s average per query on Groq free tier (occasional spikes due to queue). Trade-off: fast inference vs. free-tier limits.
- Relevance and Factuality: nitial semantic-only retrieval favored secondary reports; hybrid + metadata boost improved primary document recall. Strict prompt ensures high factuality but conservative answers when context is partial. Trade-off: stricter threshold → high factuality, but risk missing info. 
- Redundancy: Many duplicate chunks from same page -> added hash  deduplication in ingestion.
- Compliance risks: Strict prompt prevents hallucinations but may give conservative answers when context is partial (as seen in demos).

## Demo Queries Results

# With semantic retrieval
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


# With Hybrid Retrieval(BM25 + Semantic)

Running: For a medium-sized German construction company, detail the main CSRD reporting thresholds, scopes, and timelines.
Hybrid retrieved 3 chunks for query 'For a medium-sized German construction company, detail the main CSRD reporting thresholds, scopes, and timelines.'
Logged metrics for: For a medium-sized German construction company, detail the main CSRD reporting thresholds, scopes, and timelines.
Retrieved Chunks:
Score: 0.84 | Source: temanord2024-553.pdf, Page: 15
which makes it mandatory for large (>500 employees) and listed companies to
disclose sustainability information. As of January 1, 2024, large and listed
companies already subject to the NFRD will have...

Score: 0.84 | Source: temanord2024-553.pdf, Page: 15
a signiicant burden; instead, it is integrated into the annual reporting process.
Typically, the personnel responsible for the annual report are also assigned the
tasks related to the Taxonomy.
Gener...

Score: 0.84 | Source: temanord2024-553.pdf, Page: 34
25
role primarily involves reporting tasks related to the enforcement of the EU
Taxonomy. Previously, the same individual often had a broader role that included a
wider range of sustainability respons...

Metrics: Latency: 1.36s, Chunks: 3, Avg Score: 0.84
Answer: 'Insufficient information in the provided sources.'

Running: Explain EU Taxonomy alignment criteria for sustainable building renovations, including incentives for reducing embodied carbon.
Hybrid retrieved 2 chunks for query 'Explain EU Taxonomy alignment criteria for sustainable building renovations, including incentives for reducing embodied carbon.'
Logged metrics for: Explain EU Taxonomy alignment criteria for sustainable building renovations, including incentives for reducing embodied carbon.
Retrieved Chunks:
Score: 0.64 | Source: temanord2024-553.pdf, Page: 22
whether they are aligned or not with the EU taxonomy and as such are classiied as
sustainable. This is primarily due to the limited interest from both investors and
customers in aligning construction...

Score: 0.64 | Source: temanord2024-553.pdf, Page: 22
P a g e  9 | 57 
 
 
3 European context and regulatory 
framework 
At the European level, the policy framework for energy efficiency and decarbonization in the building sector 
is structured around se...

Metrics: Latency: 0.86s, Chunks: 2, Avg Score: 0.64
Answer: 'Insufficient information in the provided sources to explain EU Taxonomy alignment criteria for sustainable building renovations, including incentives for reducing embodied carbon.'

Running: Outline key GEG requirements and compliance risks for heat pump systems in new commercial buildings in Germany.
Hybrid retrieved 1 chunks for query 'Outline key GEG requirements and compliance risks for heat pump systems in new commercial buildings in Germany.'
Logged metrics for: Outline key GEG requirements and compliance risks for heat pump systems in new commercial buildings in Germany.
Retrieved Chunks:
Score: 0.63 | Source: faktenblatt-geg-gebaeudeenergiegesetz-en.pdf, Page: 0
Around three quarters of heating systems in Germany still run on 
natural gas or heating oil. In order to break this dependency, the 
amended Buildings Energy Act (GEG), in force since 1 January 
2024...

Metrics: Latency: 1.94s, Chunks: 1, Avg Score: 0.63
Answer: 'Based on the provided context, here are the key GEG requirements and compliance risks for heat pump systems in new commercial buildings in Germany:\n\n**Key GEG Requirements:**\n\n1. **Renewable energy usage**: New heating systems in new developments must use at least 65% renewable energy [Source: Not specified, but implied by the context].\n2. **Energy efficiency**: The GEG lays down energy standards for new roofs, windows, or insulated walls to ensure energy efficiency [Source: Not specified, but implied by the context].\n\n**Compliance Risks:**\n\n1. **Non-compliance with renewable energy usage**: Heat pump systems in new commercial buildings in new development areas must use at least 65% renewable energy, failing which will lead to non-compliance.\n2. **Insufficient energy efficiency**: Commercial buildings must meet energy efficiency standards for new roofs, windows, or insulated walls, non-compliance with which may lead to fines or penalties.\n\n**Insufficient information** is provided in the context regarding specific GEG requirements, such as:\n\n* Specific energy efficiency standards for new roofs, windows, or insulated walls.\n* Transitional periods for new commercial buildings outside new development areas.\n* Compliance deadlines for large cities (with more than 100,000 inhabitants) when replacing heating systems.\n\nFurther research is required to provide more detailed information on these aspects.'


NOTE:
Everything runs on free/local resources only.