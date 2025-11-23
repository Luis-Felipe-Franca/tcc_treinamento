from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from _utils import sync_photo, do_inference, gerar_pdf



##### GLOBALS #######

UPLOAD_DIR = 'uploads'

###################


os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()


app.add_middleware(CORSMiddleware,
                    allow_origins=["http://localhost:5500",
                                   "http://127.0.0.1:5500"
                                   ],
                    allow_credentials = True,
                    allow_methods=["*"],
                    allow_headers=["*"],
                    )



@app.post('/uploadPhoto')
async def upload_image(file: UploadFile = File(...),    
                       pergunta1: str = Form(...),
                       pergunta2: str = Form(...),
                       pergunta3: str = Form(...)
                       ):

    print(file.content_type)

    if file.content_type != 'image/jpeg':
        raise HTTPException(status_code =400, 
                            detail = 'O arquivo deve ser uma imagem PNG'
                            )

    photo_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(photo_path, 'wb') as f:
        f.write(await file.read())
    
    #sync_photo(photo_path)
    
    result_photo_path = do_inference(photo_path)
    
    gerar_pdf(pergunta1, 
              pergunta2, 
              pergunta3,
              result_photo_path
              )
    
    return {'message': 'Foto recebida!', 
            'filename': file.filename
            }




@app.get("/availableFiles")
async def available_files():
    directory = 'results'
    paths = [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.endswith(".png")
    ]
    return {"available_files": paths}




@app.get('/download/{filename}')
async def download_image(filename: str):
    directory = 'results'
    photo_path  = os.path.join(directory, filename)

    if not os.path.exists(photo_path):
        raise HTTPException(status_code = 404, 
                            detail = 'Arquivo não encontrado :('
                            )

    return FileResponse(photo_path,
                        media_type = 'image/png', 
                        filename = filename
                        )




# uvicorn main:app --reload     




# if __name__ == '__main__':
    
#     import uvicorn
#     uvicorn.run('main:app', port=8080, reload=True)
