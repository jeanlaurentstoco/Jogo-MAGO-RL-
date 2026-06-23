import markdown
import os

# Script para ler o markdown, adicionar CSS e MathJax e compilar como HTML.
# Isso garante que as equações matemáticas complexas (como do PPO e GAE) renderizem perfeitamente.

md_path = "Jogo_Mago_RL_Documentacao.md"
html_path = "Jogo_Mago_RL_Documentacao.html"

with open(md_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Transforma Markdown em HTML
html_content = markdown.markdown(text, extensions=['fenced_code', 'tables'])

# Injeta CSS (para simular formato A4/PDF elegante) e o Script do MathJax
html_template = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Documentação Tensor Mage Arena</title>
    <!-- Configuração do MathJax para renderizar as formulas LaTeX -->
    <script>
    MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
      }},
      svg: {{
        fontCache: 'global'
      }}
    }};
    </script>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            color: #333;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
        }}
        .page {{
            width: 210mm;
            min-height: 297mm;
            padding: 20mm;
            margin: 10mm auto;
            border: 1px #D3D3D3 solid;
            border-radius: 5px;
            background: white;
            box-shadow: 0 0 5px rgba(0, 0, 0, 0.1);
        }}
        h1, h2, h3 {{
            color: #2c3e50;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
        }}
        code {{
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 4px;
            font-family: 'Courier New', Courier, monospace;
        }}
        pre {{
            background-color: #272822;
            color: #f8f8f2;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        pre code {{
            background-color: transparent;
            color: inherit;
        }}
        @media print {{
            body, .page {{
                margin: 0;
                border: initial;
                border-radius: initial;
                width: initial;
                min-height: initial;
                box-shadow: initial;
                background: initial;
                page-break-after: always;
            }}
            .page {{
                padding: 1cm;
            }}
        }}
    </style>
</head>
<body>
    <div class="page">
        {html_content}
    </div>
</body>
</html>
"""

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_template)

print(f"[OK] Arquivo HTML com suporte a LaTeX gerado com sucesso: {html_path}")
print("Para gerar o PDF, basta abrir este HTML no navegador (ex: Chrome) e pressionar Ctrl+P (Imprimir) -> Salvar como PDF.")
print("Desta forma garantimos 100% de precisão gráfica na matemática vetorial exibida.")

try:
    from weasyprint import HTML
    pdf_path = "Jogo_Mago_RL_Documentacao_WEASYPRINT_PREVIEW.pdf"
    # Weasyprint não executa Javascript (MathJax), então as fórmulas ficam em texto crú (LaTeX bruto).
    # O HTML continua sendo a via preferida para gerar o PDF final perfeito.
    HTML(string=html_template).write_pdf(pdf_path)
    print(f"[OK] PDF de preview estático gerado (Sem JS/MathJax renderizado): {pdf_path}")
except Exception as e:
    print(f"[AVISO] Weasyprint falhou, use o HTML para salvar como PDF. Erro: {{e}}")
