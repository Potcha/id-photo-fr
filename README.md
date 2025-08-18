# ID Photo FR — V1.1 (frontend ↔ backend)
- **Frontend** : `frontend/index.html` (HTML/Canvas) appelle l’API pour `/validate` et `/mask/pipette`.
- **Backend**  : `backend/app.py` (FastAPI) avec CORS activé.

## Démarrer sous Windows (VS Code)
1. Ouvre `id-photo-fr.code-workspace`
2. Terminal → *Exécuter une tâche…* → **Backend: install deps**
3. Terminal → *Exécuter une tâche…* → **Dev: API + Frontend (servers)**
4. Ouvre `http://localhost:5173`

### Manuellement (PowerShell)
```ps1
cd backend
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m uvicorn app:app --reload --port 8000
# Nouveau terminal
cd ..\frontend
py -m http.server 5173
# http://localhost:5173
```

## Notes
- La **pipette** envoie `file + samples(x,y)` à `/mask/pipette`. L’API retourne un **PNG** alpha=masque (sujet).
- La **validation FR** (34 mm cible, 32–36 mm, yeux ±2°) est calculée côté backend (`/validate`).
- Exports : PNG 300 DPI (35×45), A4, 10×15 / 4×6.
