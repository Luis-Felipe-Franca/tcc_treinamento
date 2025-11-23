import subprocess 
from datetime import datetime
import shutil
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torchvision import transforms
from fpdf import FPDF
import os



def do_inference(photo_path: str):

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    image = Image.open(photo_path).convert("RGB")
    image_np = np.array(image)
    
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    MODEL_PATH = '../model.pth'
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        pred = model(input_tensor)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
    
    pred_np = pred.squeeze().cpu().numpy()
    
    mask_np = None
    
    overlay = np.copy(image_np)

    overlay[pred_np == 1] = [255, 0, 0]

    if mask_np is not None:
        overlay[mask_np == 1] = [255, 255, 0]
    
    alpha = 0.4
    final_image = (image_np * (1 - alpha) + overlay * alpha).astype(np.uint8)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(final_image)
    plt.title("Segmentação (Vermelho = Predição, Amarelo = Real)")
    plt.axis("off")
    
    result_path = f"results/{datetime.now().strftime('%f%d%H%Y%m%S')}.png"
    plt.savefig(result_path)
    
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

    # Carrega fonte com suporte UTF-8
    font_path = "/Library/Fonts/Arial Unicode.ttf"
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Fonte não encontrada: {font_path}")
    
    pdf.add_font("ArialUni", "", font_path, uni=True)
    pdf.set_font("ArialUni", "", 13)

    # Estilo
    primary = (40, 60, 120)
    secondary = (240, 240, 245)
    text = (50, 50, 50)

    pdf.set_fill_color(*secondary)
    pdf.rect(0, 0, 210, 297, "F")

    # Título
    pdf.set_text_color(*primary)
    pdf.set_font("ArialUni", "", 20)
    pdf.cell(0, 12, "Relatório de Respostas", ln=True, align="C")
    pdf.ln(4)

    pdf.set_draw_color(*primary)
    pdf.set_line_width(1)
    pdf.line(30, 28, 180, 28)
    pdf.ln(8)

    pdf.set_font("ArialUni", "", 13)
    pdf.set_text_color(*text)

    # Função bloca perguntas
    def bloco(pergunta, texto):
        pdf.set_font("ArialUni", "", 14)
        pdf.set_text_color(*primary)
        pdf.cell(0, 8, f"• {pergunta}", ln=True)

        pdf.set_font("ArialUni", "", 12)
        pdf.set_text_color(*text)
        pdf.multi_cell(0, 7, texto)
        pdf.ln(3)

    bloco("Pergunta 1:", pergunta1)
    bloco("Pergunta 2:", pergunta2)
    bloco("Pergunta 3:", pergunta3)

    # Imagem
    pdf.set_font("ArialUni", "", 14)
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
    pdf.set_font("ArialUni", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 10, "Gerado automaticamente • © 2025", align="C")

    pdf.output(pdf_path)