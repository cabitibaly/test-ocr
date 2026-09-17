from fastapi import FastAPI, UploadFile, File
from paddleocr import PaddleOCR
import tempfile
import os
import re
from typing import List

app = FastAPI()

ocr = PaddleOCR(
    lang='fr',    
    use_textline_orientation=True,
    enable_mkldnn=False
)

_MRZ_FORMATS = {
    3: 30,
    2: 44,
}

_MRZ_CHARSET = re.compile(r"^[A-Z0-9<]+$")


def extract_mrz(texts: List[str]) -> List[str]:
    cleaned = [t.strip() for t in texts]
    n = len(cleaned)

    for line_count, line_len in _MRZ_FORMATS.items():
        if n < line_count:
            continue
        window = cleaned[-line_count:]
        if all(
            len(line) == line_len and _MRZ_CHARSET.match(line)
            for line in window
        ):
            return window

    return []

def extract_passeport_data(texts: list):
    if len(texts) < 2:
        return {
            "error": "Length of extracted MRZ lines is less than 2."
        }
    
    texts = [t.strip() for t in texts]
    i = next((idx for idx, t in enumerate(texts) if t.startswith("P")), None)
    
    if i is None:
        return {
            "error": "MRZ not found in the provided texts."    
        }
        
    line1, line2 = texts[i], texts[i + 1]
    
    doc_type = line1[:2].rsplit("<", 1)[0]
    print(doc_type)
    
    nom, prenoms = line1[5:].split("<<", 1)
    nom = nom.replace("<", " ").strip()
    prenoms = prenoms.replace("<", " ").strip()
    
    numero_passport = line2[:9].rsplit("<")[0]
    nationalite = line2[10:13]
    date_naissance = line2[13:19]
    jour = date_naissance[4:6]
    mois = date_naissance[2:4]
    annee = date_naissance[:2]
    sex = line2[20]   
    date_expiration = line2[21:27] 
    
    day_exp = date_expiration[4:6]
    month_exp = date_expiration[2:4]
    year_exp = date_expiration[:2]
    
    return {
        "document_type": "Passport" if doc_type.startswith("P") else "Unknown",
        "nationalite": nationalite,
        "numero_passeport": numero_passport,
        "date_naissance": f"{jour}/{mois}/{annee}",
        "date_expiration": f"{day_exp}/{month_exp}/{year_exp}",
        "sex": sex,
        "nom": nom,
        "prenoms": prenoms
    }

def extract_cnib_data(texts: list):    
    if len(texts) < 3:
        return {
            "error": "Length of extracted MRZ lines is less than 3."
        }
    
    texts = [t.strip() for t in texts]
    i = next((idx for idx, t in enumerate(texts) if t.startswith("I")), None)
    
    if i is None:
        return {
            "error": "MRZ not found in the provided texts."    
        }
        
    line1, line2, line3 = texts[i], texts[i + 1], texts[i + 2]
            
    doc_type = line1[:2].rsplit("<", 1)[0]
    numero_cnib = line1[5:14].rsplit("<")[0]
            
    date_naissance = line2[:6]
    jour = date_naissance[4:6]
    mois = date_naissance[2:4]
    annee = date_naissance[:2]
    sexe = line2[7]
    
    date_expiration = line2[8:14]
    day_exp = date_expiration[4:6]
    month_exp = date_expiration[2:4]
    year_exp = date_expiration[:2]
    
    nationalite = line2[15:18]
            
    nom, prenoms = line3.split("<<", 1)
    nom = nom.replace("<", " ").strip()
    prenoms = prenoms.replace("<", " ").strip()
            
    return {
        "document_type": "CNI" if doc_type.startswith("I") else "Unknown",
        "nationalite": nationalite,
        "numero_cnib": numero_cnib,
        "date_naissance": f"{jour}/{mois}/{annee}",
        "date_expiration": f"{day_exp}/{month_exp}/{year_exp}",
        "sexe": sexe,
        "nom": nom,
        "prenoms": prenoms
    }                


@app.post("/ocr")
async def process_ocr(file: UploadFile = File(...)):    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp.write(await file.read())
        image_path = temp.name
        
    try:        
        result = ocr.predict(image_path)

        texts = []
        scores = []        

        for res in result:
            data = res.json["res"]            
            texts.extend(data.get("rec_texts", []))            
            scores.extend(data.get("rec_scores", []))        

        mrz_lines = extract_mrz(texts)        
        # return {
        #     "texts": texts,  
        #     "scores": scores
        # }
        
        if len(mrz_lines) == 0:
            return {
                "error": "No MRZ found.",
                "texts": texts,
                "scores": scores
            }
        
        if mrz_lines[0].startswith("P"):            
            return extract_passeport_data(mrz_lines)
        elif mrz_lines[0].startswith("I"):            
            return extract_cnib_data(mrz_lines)
        else:
            return {
                "error": "No card detected.",
                "texts": texts,
                "scores": scores
            }

    finally:        
        os.remove(image_path)    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)