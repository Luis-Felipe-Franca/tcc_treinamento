from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import os
from _utils import sync_photo


##### GLOBALS #######

UPLOAD_DIR = 'uploads'

###################


os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()




@app.post('/uploadPhoto')
async def upload_image(file: UploadFile = File(...)):

    if file.content_type != 'image/png':
        raise HTTPException(status_code =400, 
                            detail = 'O arquivo deve ser uma imagem PNG'
                            )

    photo_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(photo_path, 'wb') as f:
        f.write(await file.read())
    
    sync_photo(photo_path)
    
    return {'message': 'Foto recebida!', 
            'filename': file.filename
            }



@app.get('/download/{filename}')
async def download_image(filename: str):
    photo_path  = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(photo_path):
        raise HTTPException(status_code = 404, 
                            detail = 'Arquivo não encontrado :('
                            )

    return FileResponse(photo_path,
                        media_type = 'image/png', 
                        filename = filename
                        )







# if __name__ == '__main__':
    
#     import uvicorn
#     uvicorn.run('main:app', port=8080, reload=True)
