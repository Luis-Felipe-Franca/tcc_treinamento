import subprocess 
from datetime import datetime
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from fpdf import FPDF
import os
from ultralytics import YOLO



def do_inference(photo_path: str):

    model = YOLO('best.pt')
    
    results = model.predict(source=photo_path, conf=0.3)
    
    result = results[0]
    
    image = Image.open(photo_path).convert("RGB")
    image_np = np.array(image)
    
    annotated_image = result.plot()
    
    if annotated_image is not None:
        final_image = annotated_image[..., ::-1]
    else:
        final_image = image_np
    
    plt.figure(figsize=(10, 5))
    plt.imshow(final_image)
    plt.title("Detecção YOLO")
    plt.axis("off")
    
    result_path = f"results/{datetime.now().strftime('%f%d%H%Y%m%S')}.png"
    plt.savefig(result_path)
    plt.close()
    
    return result_path


def git_pull():
    
    subprocess.run(['git', 'pull'],
                    check = True,
                    capture_output = True,
                    text = True
                    )    
     

def git_add(file_path:str):
    
    subprocess.run(['git', 'add', file_path],
                    check = True,
                    capture_output = True,
                    text = True
                    )        
    

def commit_and_push(file_path:str, commit_message:str):

    
    subprocess.run(['git', 'commit', '-m', f'"{commit_message}"', file_path],
                    check = True,
                    capture_output = True,
                    text = True
                    )   
    
    print(file_path)

    subprocess.run(['git', 'push'],
                    check = True, 
                    capture_output = True,
                    text = True
                    )
    

def sync_photo(photo_path:str):
    
    git_pull()
    
    new_path = f"recived_photos/{datetime.now().strftime('%f%d%H%Y%m%M%S')}.png"
    shutil.copy(photo_path, new_path)
    
    git_add(new_path)
    path = do_inference(photo_path)

    commit_and_push(file_path = path, 
                    commit_message = 'New photo uploaded from API!'
                    )
    
    
def gerar_pdf(pergunta1:str,
              pergunta2:str, 
              pergunta3:str, 
              image_path:str,
              pdf_path="results/resultado.pdf"
              ):

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Usa apenas 'fonts/DejaVuSans.ttf' dentro da pasta `api\fonts`.
    dejavu_path = os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf")
    
    pdf.add_font("AppFont", "", dejavu_path, uni=True)
    pdf.set_font("AppFont", "", 13)

    # Estilo
    primary = (40, 60, 120)
    secondary = (240, 240, 245)
    text = (50, 50, 50)

    pdf.set_fill_color(*secondary)
    pdf.rect(0, 0, 210, 297, "F")

    # Título
    pdf.set_text_color(*primary)
    pdf.set_font("AppFont", "", 20)
    pdf.cell(0, 12, "Relatório de Respostas", ln=True, align="C")
    pdf.ln(4)

    pdf.set_draw_color(*primary)
    pdf.set_line_width(1)
    pdf.line(30, 28, 180, 28)
    pdf.ln(8)

    pdf.set_font("AppFont", "", 13)
    pdf.set_text_color(*text)

    # Função bloca perguntas
    def bloco(pergunta, texto):
        pdf.set_font("AppFont", "", 14)
        pdf.set_text_color(*primary)
        pdf.cell(0, 8, f"• {pergunta}", ln=True)

        pdf.set_font("AppFont", "", 12)
        pdf.set_text_color(*text)
        pdf.multi_cell(0, 7, texto)
        pdf.ln(3)

    bloco("Pergunta 1:", pergunta1)
    bloco("Pergunta 2:", pergunta2)
    bloco("Pergunta 3:", pergunta3)

    # Imagem
    pdf.set_font("AppFont", "", 14)
    pdf.set_text_color(*primary)
    pdf.cell(0, 8, "Imagem enviada:", ln=True)
    pdf.ln(2)

    try:
        pdf.set_draw_color(180, 180, 180)
        y = pdf.get_y()
        pdf.rect(25, y, 160, 120)
        pdf.image(image_path, x=30, y=y+3, w=150)
    except Exception as e:
        pdf.set_text_color(255, 0, 0)
        pdf.multi_cell(0, 8, f"Erro ao carregar imagem:\n{e}")

    # Rodapé
    pdf.set_y(-15)
    pdf.set_font("AppFont", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 10, "Gerado automaticamente • © 2025", align="C")

    pdf.output(pdf_path)