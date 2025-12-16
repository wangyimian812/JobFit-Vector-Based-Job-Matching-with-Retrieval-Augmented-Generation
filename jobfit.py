import pandas as pd
import numpy as np
import faiss #library that quickly finds vectors that are most similar to another vector
from sentence_transformers import SentenceTransformer
import ollama

# ---------- PROFILE ----------
# This is my personal profile that will be compared against job ads
my_skills = [
    "python",
    "sql",
    "html",
    "css",
    "javascript",
    "flask",
    "node.js",
    "git",
    "github",
    "oracle service cloud",
    "oracle integration cloud",
    "sqlalchemy",
    "bootstrap",
    "ajax",
    "c programming",
    "java",
    "web development",
    "database design",
    "software testing basics"
]

my_level = "junior"

def ask_llm(context): # Build a strict prompt for LLM to use
    prompt = (
        "You are a job analysis assistant.\n"
        "Rules:\n"
        "- Use ONLY the retrieved job chunks.\n"
        "- Do NOT guess or infer anything.\n"
        "- Every sentence MUST cite a chunk like [Chunk 1] or [Chunk 2].\n"
        "- If you cannot support something with a chunk, say: Not stated in chunks.\n\n"
        + context +
        "\nOutput format:\n"
        "Reasoning:\n"
        "- <sentence> [Chunk X]\n"
        "- <sentence> [Chunk Y]\n"
        "- <sentence> [Chunk Z]\n"
    )

    r = ollama.chat(
        model="gemma3:4b", #The model I am using on Ollama (a local LLM)
        messages=[{"role": "user", "content": prompt}]
    )
    return r["message"]["content"]  #LLM's response text

def chunk_text(text, size=800, overlap=200): #Chunking
    text = " ".join(text.split()) #Clean up whitespace

    chunks = []
    i = 0
    step = size - overlap

    while i < len(text): #Split long job text into overlapping chunks
        chunks.append(text[i:i+size])
        i += step
    return chunks


def build_profile_text(): # the text combination will be embedded into a vector
    return (
        "skills: " + " ".join(my_skills) +
        ". level: " + my_level
    )

def build_job_text(row):
    return ( # what is embeded
        str(row["job_title"]) + " " +
        str(row["job_description"]) + " " +
        str(row["work_rights_requirement"])
    )

def is_senior_job(job_title):
    t = str(job_title).lower()
    return ("senior" in t) or ("lead" in t) or ("principal" in t)

def decide_apply(my_level, job_title):
    if my_level == "junior" and is_senior_job(job_title):
        return "Senior job, not applicable"
    return "Apply"

def work_rights_decision(work_rights_text):
    work_rights_text_lower = str(work_rights_text).lower()

    if "citizen" in work_rights_text_lower:
        return "Not applicable (Australian citizen required)"

    if "permanent resident" in work_rights_text_lower or "pr only" in work_rights_text_lower:
        return "Not applicable (Permanent resident required)"

    if "sponsorship not available" in work_rights_text_lower:
        return "Not applicable (No visa sponsorship)"

    return "Applicable!"

def government_role_decision(job_text):
    text_lower = str(job_text).lower()
    keywords = ["federal government", "government role", "department", "aps", "defence", "nv1", "nv2", "baseline clearance"]
   
    for k in keywords:
        if k in text_lower:
            return "Not applicable (Government / clearance role)"
        
    return "Applicable!"

def ingest_jobs():
    # Load job ads from CSV file 
    return pd.read_csv("jobs.csv")

def run_jobfit():
    jobs = ingest_jobs()

    job_texts = [] #All text chunks
    chunk_job_rows = [] #Mappling from chunk to original job

    for _, row in jobs.iterrows(): #Loop through each job ad
        job_id_for_key = "" if pd.isna(row.get("job_id")) else str(row.get("job_id"))
        company_for_key = "" if pd.isna(row.get("company_name")) else str(row.get("company_name"))
        title_for_key = str(row.get("job_title", ""))

        if job_id_for_key != "":
            job_key = job_id_for_key
        else:
            job_key = title_for_key + "||" + company_for_key

        for chunking in chunk_text(build_job_text(row)): #Chunk each job and store them
            job_texts.append(chunking)
            chunk_job_rows.append({"row": row, "job_key": job_key})

    model = SentenceTransformer("all-MiniLM-L6-v2") #384 dimensions

    job_vectors = model.encode(job_texts, normalize_embeddings=True) #Convert chunks into vectors
    job_vectors = np.array(job_vectors, dtype="float32")

    dimension = job_vectors.shape[1] 

    index = faiss.IndexFlatIP(dimension)
    index.add(job_vectors)

    profile_vector = model.encode( #Convert profile into a vector
        [build_profile_text()],
        normalize_embeddings=True
    )
    profile_vector = np.array(profile_vector, dtype="float32")

    similarities, indices = index.search(profile_vector, len(job_texts)) #Search for most similar job chunks

    results = []
    seen = set()
    for rank, idx in enumerate(indices[0]):
        row = chunk_job_rows[idx]["row"]
        job_key = chunk_job_rows[idx]["job_key"]

        job_id = "" if pd.isna(row.get("job_id")) else row.get("job_id")
        company = "" if pd.isna(row.get("company_name")) else row.get("company_name")
        title = row.get("job_title", "")

        key = (job_id, title, company)
        if key in seen:
            continue
        seen.add(key)

        profile_to_job_semantic_relevance = similarities[0][rank]

        reasons = []
        if decide_apply(my_level, row["job_title"]) != "Apply":
            reasons.append("Senior job")

        rights = work_rights_decision(row["work_rights_requirement"])
        if rights.startswith("Not applicable"):
            reasons.append(rights.replace("Not applicable (", "").replace(")", ""))

        gov = government_role_decision(
            str(row["job_title"]) + " " + str(row["job_description"])
        )
        if gov.startswith("Not applicable"):
            reasons.append("Government / clearance role")

        if len(reasons) == 0:
            decision = "Apply"
        else:
            decision = "Not applicable (" + " + ".join(reasons) + ")"

        location = "" if pd.isna(row.get("job_location")) else row.get("job_location")

        results.append({
            "rank": rank + 1,
            "job_id": job_id,
            "job_key": job_key,
            "job_title": row.get("job_title", ""),
            "company_name": company,
            "job_location": location,
            "profile_to_job_semantic_relevance": round(float(profile_to_job_semantic_relevance), 3),
            "decision": decision,
            "work_rights": rights
        })

    return results, job_texts, indices, chunk_job_rows


def main():
    results, job_texts, indices, chunk_job_rows = run_jobfit()

    print("\nJob matches:\n")
    for r in results:
        print("----")
        print("Job ID:", r["job_id"])
        print("Title:", r["job_title"])
        print("Company:", r["company_name"])
        print("Location:", r["job_location"])
        print("profile_to_job_semantic_relevance:", r["profile_to_job_semantic_relevance"])
        print("Decision:", r["decision"])

        if str(r["work_rights"]).startswith("Not applicable"):
            print("Work rights:", r["work_rights"])
        else:
            print("Work rights: Applicable!")

    best_job = None
    for r in results:
        if r["decision"] == "Apply":
            best_job = r
            break

    retrieved_chunks = [] #Retrieve top chunks for that best job
    top_k = 5
    best_key = best_job["job_key"]

    for idx in indices[0]:
        if chunk_job_rows[idx]["job_key"] == best_key:
            retrieved_chunks.append(job_texts[idx])
        if len(retrieved_chunks) >= top_k:
            break

    context = ""
    company = best_job["company_name"]
    if company == "":
        company = "Company info not provided in job ad"

    context += "Job title: " + str(best_job["job_title"]) + "\n"
    context += "Company: " + str(company) + "\n"
    context += "Decision: " + str(best_job["decision"]) + "\n"
    context += "Relevance score: " + str(best_job["profile_to_job_semantic_relevance"]) + "\n\n"
    context += "Retrieved job chunks:\n"
    for i, chunk in enumerate(retrieved_chunks, 1):
        context += "[Chunk " + str(i) + "]\n"
        context += chunk + "\n\n"

    answer = ask_llm(context)

    print("\nLLM answer:")
    print(answer)


if __name__ == "__main__":
    main()
