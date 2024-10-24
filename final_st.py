import os
import streamlit as st
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Image
from PIL import Image as PILImage
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from reportlab.lib.pagesizes import letter, inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PlatypusImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO

# Function to process the PDF file
def process_pdf(pdf_file_path, output_dir):
    # Extract elements from the PDF using partition_pdf function
    elements = partition_pdf(
        filename=pdf_file_path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_output_dir=output_dir
    )

    # Initialize the Ollama LLM with the Llama2:7b model
    llm = ChatOllama(model="llama2:7b")

    # Prepare the content for the PDF
    display_content = []

    # Process elements in batches
    batch_size = 23
    for i in range(0, len(elements), batch_size):
        batch = elements[i:i + batch_size]

        # Collect text and images in order
        batch_text = []
        batch_images = []

        for element in batch:
            if isinstance(element, Image):
                image_path = element.metadata.image_path if element.metadata else None
                if image_path and os.path.exists(image_path):
                    batch_images.append(image_path)
            else:
                element_str = str(element).strip()
                if element_str:
                    batch_text.append(element_str)

        # Combine all text into one batch and process it
        if batch_text:
            text = "\n\n".join(batch_text)

            # Create a prompt for the LLM summarization
            prompt = PromptTemplate(template="Summarize the following text: {text}", input_variables=["text"])
            chain = prompt | llm

            # Send the prompt to the LLM and get the summary
            response = chain.invoke({"text": text})
            summary = response.content.strip()
            display_content.append({"type": "text", "content": summary})

        # Store images after processing text for each batch
        if batch_images:
            for image_path in batch_images:
                display_content.append({"type": "image", "path": image_path})

    return display_content

# PDF Generation Function
def generate_summary_pdf(display_content, output_pdf):
    # Create a buffer for the PDF
    buffer = BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=inch, rightMargin=inch)

    # Prepare content with correct formatting
    styles = getSampleStyleSheet()
    custom_style = ParagraphStyle(
        'Custom',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=10,
        leading=12,
    )

    flowables = []

    for element in display_content:
        if element["type"] == "text":
            paragraphs = element["content"].split("\n\n")
            for paragraph in paragraphs:
                p = Paragraph(paragraph, custom_style)
                flowables.append(p)
                flowables.append(Spacer(1, 12))  # Add space between paragraphs
        elif element["type"] == "image":
            # Add the image to the PDF with proper padding and scaling
            try:
                img = PlatypusImage(element["path"])
                img.drawHeight = 3 * inch  # Set max height
                img.drawWidth = 4 * inch  # Set max width
                img.hAlign = 'CENTER'  # Center the image on the page
                flowables.append(img)
                flowables.append(Spacer(1, 12))  # Add space after the image
            except Exception as e:
                print(f"Error adding image {element['path']}: {e}")

    # Build the PDF
    pdf.build(flowables)

    # Save PDF content
    buffer.seek(0)
    with open(output_pdf, "wb") as f:
        f.write(buffer.getvalue())

# Streamlit Frontend
def main():
    st.title("PDF Summarizer with LLM")
    st.write("Upload a PDF file to generate a summary along with images.")

    # Upload PDF file
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        # Save the uploaded file
        input_pdf = "uploaded_pdf.pdf"
        with open(input_pdf, "wb") as f:
            f.write(uploaded_file.read())

        # Create output directory
        output_dir = "extracted_content"
        os.makedirs(output_dir, exist_ok=True)

        # Process the PDF
        with st.spinner("Processing the PDF..."):
            content = process_pdf(input_pdf, output_dir)

        # Generate summary PDF
        output_pdf = "summary.pdf"
        generate_summary_pdf(content, output_pdf)

        # Provide download link for the generated PDF
        with open(output_pdf, "rb") as f:
            st.download_button(label="Download Summary PDF", data=f, file_name="summary.pdf", mime="application/pdf")

        st.success("Summary PDF generated successfully!")

if __name__ == "__main__":
    main()
