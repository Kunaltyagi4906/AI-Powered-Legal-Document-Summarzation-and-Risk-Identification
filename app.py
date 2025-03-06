import streamlit as st
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from fpdf import FPDF
import time

load_dotenv()

# Initialize session state variables
if "default_model" not in st.session_state:
    st.session_state["default_model"] = "llama-3.3-70b-versatile"
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "final_summary" not in st.session_state:
    st.session_state["final_summary"] = ""
if "risk_assessment" not in st.session_state:
    st.session_state["risk_assessment"] = ""
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []

tab1, tab2, tab3 = st.tabs(["Summary", "QnA ChatBot","Risk Assessment"])

with tab1:
    st.title("Experiment Legal Document Analyzer")
    st.divider()

    st.markdown("## Summarize and assess risks in your legal documents")

    # Upload File
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "csv"])

    llm = ChatGroq(model="mixtral-8x7b-32768")
    parser = StrOutputParser()
    prompt_template = ChatPromptTemplate.from_template("Summarize the Following Document {document}")
    chain = prompt_template | llm | parser

    if uploaded_file is not None:
        with st.spinner("Processing..."):
            try:
                print("File: ", uploaded_file)
                print("File type: ", uploaded_file.type)
                
                temp_file_path = uploaded_file.name
                print("File path: ", temp_file_path)
                
                # Save uploaded file
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Create document loader
                if uploaded_file.type == "text/plain":
                    loader = TextLoader(temp_file_path)
                elif uploaded_file.type == "text/csv":
                    loader = CSVLoader(temp_file_path)
                elif uploaded_file.type == "application/pdf":
                    loader = PyPDFLoader(temp_file_path)
                else:
                    st.error("File type is not supported!")
                    st.stop()
                    
                # Create the document    
                doc = loader.load()
                print(doc)
                
                # Text Splitter - make chunks smaller to avoid token limits
                text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                
                # Split the document into chunks
                st.session_state["chunks"] = text_splitter.split_documents(doc)
                print(f"Number of chunks: {len(st.session_state['chunks'])}")
            
            except Exception as e:
                print(e)
                st.error(f"Error processing file: {e}")
        st.success("File Uploaded")

    # Summary of Document
    if st.button("Summarize"):
        if "chunks" in st.session_state and st.session_state["chunks"]:
            summary_container = st.empty()
            chunk_summaries = []
            
            with st.spinner("Analyzing document chunks..."):
                progress_bar = st.progress(0)
                total_chunks = len(st.session_state["chunks"])
                
                try:
                    for i, chunk in enumerate(st.session_state["chunks"]):
                        # Summary prompt
                        chunk_prompt = ChatPromptTemplate.from_template(
                            "You are a highly skilled legal expert tasked with summarizing legal text. "
                            "Please summarize the following chunk of legal text in a concise manner, "
                            "highlighting the most critical information. Focus on key clauses, obligations, "
                            "rights, and definitions. Do not omit any key details:\n\n{document}"
                        )
                        
                        # Chain for summary
                        chunk_chain = chunk_prompt | llm | parser
                        chunk_summary = chunk_chain.invoke({"document": chunk.page_content})
                        chunk_summaries.append(chunk_summary)
                        
                        # Update progress
                        progress_bar.progress((i + 1) / total_chunks)
                        
                        # Add a small delay to avoid rate limiting
                        time.sleep(0.5)
                        
                except Exception as e:
                    print("Error analyzing chunks", e)
                    st.error(f"Error analyzing chunks: {e}")
                    st.stop()
                    
            # Final summary
            with st.spinner("Creating final summary..."):
                try:
                    # Process summaries in batches to stay within token limits
                    batch_size = 5  # Adjust based on your content size
                    batched_summaries = []
                    
                    for i in range(0, len(chunk_summaries), batch_size):
                        batch = chunk_summaries[i:i+batch_size]
                        combined_batch = "\n\n".join(batch)
                        
                        # Create intermediate summary prompt
                        intermediate_summary_prompt = ChatPromptTemplate.from_template(
                            "You are a senior legal expert tasked with creating a concise summary from these "
                            "document summaries. Combine these summaries into a cohesive and "
                            "comprehensive summary:\n\n{document}"
                        )
                        
                        # Generate intermediate summary
                        intermediate_summary_chain = intermediate_summary_prompt | llm | parser
                        intermediate_summary = intermediate_summary_chain.invoke({"document": combined_batch})
                        batched_summaries.append(intermediate_summary)
                        
                        # Add a small delay to avoid rate limiting
                        time.sleep(1)
                    
                    # Final combination of batched summaries
                    combined_batched_summaries = "\n\n".join(batched_summaries)
                    
                    # Create final summary prompt
                    final_summary_prompt = ChatPromptTemplate.from_template(
                        "You are a senior legal expert tasked with creating a final summary from summarized chunks "
                        "of a legal document. Combine the key points from the provided summaries into a cohesive and "
                        "comprehensive summary. The final summary should be organized by key sections (e.g., Parties, "
                        "Definitions, Obligations, Rights, Term & Termination, etc.) and be detailed enough to capture "
                        "the main legal implications:\n\n{document}"
                    )
                    
                    # Generate final summary
                    final_summary_chain = final_summary_prompt | llm | parser
                    final_summary = final_summary_chain.invoke({"document": combined_batched_summaries})
                    
                    # Store results in session state
                    st.session_state["final_summary"] = final_summary
                    
                    # Display results
                    st.subheader("Document Summary")
                    st.write(final_summary)
                    
                    def create_summary_pdf(summary_content):
                        pdf = FPDF()
                        
                        # Add summary page
                        pdf.add_page()
                        pdf.set_font("Arial", "B", 16)
                        pdf.cell(0, 10, "Legal Document Summary", ln=True, align="C")
                        pdf.ln(10)
                        
                        pdf.set_font("Arial", size=12)
                        lines = summary_content.split('\n')
                        for line in lines:
                            # Replace any non-latin1 characters
                            safe_line = ''.join(c if ord(c) < 256 else ' ' for c in line)
                            pdf.multi_cell(0, 10, txt=safe_line)
                        
                        return pdf.output(dest="S").encode("latin-1")
                    
                    # Download summary report
                    summary_pdf_data = create_summary_pdf(final_summary)
                    st.download_button(
                        label="Download Summary",
                        data=summary_pdf_data,
                        file_name="legal_document_summary.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    print("Error creating final summary", e)
                    st.error(f"Error creating final summary: {e}")
        else:
            st.error("Please upload a file first!")

with tab2:
    # heading
    st.title("Legal Document QnA")
    st.write("Ask questions about your legal document and its risks")
    st.divider()

    # Display chat messages
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    user_question = st.chat_input("Ask a question about your legal document:")
    
    if user_question:
        # Add user message to chat history
        st.session_state["messages"].append({"role": "user", "content": user_question})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_question)
            
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                if st.session_state["final_summary"] and st.session_state["risk_assessment"]:
                    # Combine summary and risk assessment as context
                    context = f"DOCUMENT SUMMARY:\n{st.session_state['final_summary']}\n\nRISK ASSESSMENT:\n{st.session_state['risk_assessment']}"
                    
                    # Create prompt with context
                    chat_prompt = ChatPromptTemplate.from_template(
                        "You are a legal assistant answering questions about a legal document. "
                        "Use the following summary and risk assessment as context to answer the question "
                        "with a cautious, precise legal perspective.\n\n"
                        "CONTEXT: {context}\n\n"
                        "QUESTION: {question}\n\n"
                        "If the answer isn't clearly in the context, acknowledge that limitation and provide "
                        "general legal information with appropriate disclaimers. Avoid making definitive "
                        "legal conclusions without sufficient information."
                    )
                    chat_chain = chat_prompt | llm | parser
                    response = chat_chain.invoke({
                        "context": context,
                        "question": user_question
                    })
                else:
                    response = "Please analyze a legal document in the Summary tab and generate a risk assessment in the Risk Assessment tab first."
                
                st.write(response)
                
        # Add assistant response to chat history
        st.session_state["messages"].append({"role": "assistant", "content": response})

with tab3:
    # Risk Assessment tab
    st.title("Legal Document Risk Assessment")
    st.write("Generate a detailed risk assessment for your legal document")
    st.divider()
    
    # Check if document has been uploaded
    if "chunks" not in st.session_state or not st.session_state["chunks"]:
        st.warning("Please upload a document in the Summary tab first.")
    else:
        # Risk Assessment Generation
        if st.button("Generate Risk Assessment"):
            with st.spinner("Analyzing risks in document chunks..."):
                try:
                    chunk_risks = []
                    
                    # Display progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    total_chunks = len(st.session_state["chunks"])
                    
                    for i, chunk in enumerate(st.session_state["chunks"]):
                        status_text.text(f"Processing chunk {i+1} of {total_chunks}")
                        
                        # Risk assessment prompt
                        risk_prompt = ChatPromptTemplate.from_template(
                            "You are a legal risk assessment expert. Review the following chunk of legal text "
                            "and identify potential legal risks, including but not limited to:\n"
                            "1. Ambiguous or vague terms\n"
                            "2. Compliance gaps or regulatory issues\n"
                            "3. Contradictory clauses\n"
                            "4. Missing essential terms\n"
                            "5. Unfavorable indemnification clauses\n"
                            "6. Liability exposures\n"
                            "7. Termination vulnerabilities\n\n"
                            "For each identified risk, specify the clause or section, explain the issue, "
                            "and provide a severity assessment (Low, Medium, High).\n\n"
                            "TEXT TO ANALYZE:\n{document}"
                        )
                        
                        # Chain for risk assessment
                        risk_chain = risk_prompt | llm | parser
                        chunk_risk = risk_chain.invoke({"document": chunk.page_content})
                        chunk_risks.append(chunk_risk)
                        
                        # Update progress
                        progress_bar.progress((i + 1) / total_chunks)
                        
                        # Add a delay to avoid rate limiting
                        time.sleep(1)
                        
                except Exception as e:
                    print("Error analyzing risks", e)
                    st.error(f"Error analyzing risks: {e}")
                    st.stop()
                    
                # Create final risk assessment
                with st.spinner("Creating final risk assessment..."):
                    try:
                        # Process risks in batches to stay within token limits
                        batch_size = 3  # Use a smaller batch size for risk assessments
                        batched_risks = []
                        
                        status_text.text("Processing risk batches...")
                        batch_progress = st.progress(0)
                        num_batches = (len(chunk_risks) + batch_size - 1) // batch_size
                        
                        for i in range(0, len(chunk_risks), batch_size):
                            batch = chunk_risks[i:i+batch_size]
                            combined_batch = "\n---\n".join(batch)
                            
                            # Create intermediate risk assessment prompt
                            intermediate_risk_prompt = ChatPromptTemplate.from_template(
                                "You are a legal risk assessment expert tasked with creating a concise risk report "
                                "from individual risk assessments. Combine these risk assessments, eliminate duplicates, "
                                "and prioritize by severity:\n\n{document}"
                            )
                            
                            # Generate intermediate risk assessment
                            intermediate_risk_chain = intermediate_risk_prompt | llm | parser
                            intermediate_risk = intermediate_risk_chain.invoke({"document": combined_batch})
                            batched_risks.append(intermediate_risk)
                            
                            # Update batch progress
                            batch_progress.progress((i + batch_size) / len(chunk_risks))
                            
                            # Add a delay to avoid rate limiting
                            time.sleep(1.5)
                        
                        # Process batched risks in smaller groups if there are many
                        if len(batched_risks) > 2:
                            status_text.text("Combining intermediate risk assessments...")
                            secondary_batched_risks = []
                            secondary_batch_size = 2
                            
                            for i in range(0, len(batched_risks), secondary_batch_size):
                                batch = batched_risks[i:i+secondary_batch_size]
                                combined_batch = "\n\n".join(batch)
                                
                                # Create secondary intermediate risk prompt
                                secondary_risk_prompt = ChatPromptTemplate.from_template(
                                    "You are a legal risk assessment expert. Combine these risk assessments "
                                    "into a single cohesive report, eliminating duplicates and organizing by category:\n\n{document}"
                                )
                                
                                # Generate secondary intermediate risk assessment
                                secondary_risk_chain = secondary_risk_prompt | llm | parser
                                secondary_risk = secondary_risk_chain.invoke({"document": combined_batch})
                                secondary_batched_risks.append(secondary_risk)
                                
                                # Add a delay to avoid rate limiting
                                time.sleep(1.5)
                            
                            combined_risks = "\n\n".join(secondary_batched_risks)
                        else:
                            # Combine batched risks directly if there are few enough
                            combined_risks = "\n\n".join(batched_risks)
                        
                        status_text.text("Creating final risk assessment report...")
                        
                        # Create final risk assessment prompt
                        final_risk_prompt = ChatPromptTemplate.from_template(
                            "You are a legal risk assessment expert tasked with creating a comprehensive risk report "
                            "from risk assessments of document chunks. Create a consolidated risk report "
                            "that categorizes and prioritizes the identified risks. Focus on:\n"
                            "1. Group related risks by category (e.g., Compliance, Liability, Ambiguity)\n"
                            "2. Prioritize risks by severity\n"
                            "3. Provide specific mitigation recommendations\n\n"
                            "Format the report with clear sections, bullet points for individual risks, and "
                            "a summary risk profile (Low, Medium, or High) for the overall document.\n\n"
                            "INPUT RISK ASSESSMENTS:\n{document}"
                        )
                        
                        # Generate final risk assessment
                        final_risk_chain = final_risk_prompt | llm | parser
                        final_risk_assessment = final_risk_chain.invoke({"document": combined_risks})
                        
                        # Store results in session state
                        st.session_state["risk_assessment"] = final_risk_assessment
                        
                        # Hide progress indicators
                        status_text.empty()
                        
                        # Display results
                        st.subheader("Risk Assessment")
                        st.write(final_risk_assessment)
                        
                        # Create PDF function for risk assessment
                        def create_risk_pdf(risk_content):
                            pdf = FPDF()
                            
                            # Add risk assessment page
                            pdf.add_page()
                            pdf.set_font("Arial", "B", 16)
                            pdf.cell(0, 10, "Risk Assessment Report", ln=True, align="C")
                            pdf.ln(10)
                            
                            pdf.set_font("Arial", size=12)
                            risk_lines = risk_content.split('\n')
                            for line in risk_lines:
                                # Replace any non-latin1 characters
                                safe_line = ''.join(c if ord(c) < 256 else ' ' for c in line)
                                pdf.multi_cell(0, 10, txt=safe_line)
                                
                            return pdf.output(dest="S").encode("latin-1")
                        
                        # Download risk assessment report
                        risk_pdf_data = create_risk_pdf(final_risk_assessment)
                        st.download_button(
                            label="Download Risk Assessment",
                            data=risk_pdf_data,
                            file_name="legal_document_risk_assessment.pdf",
                            mime="application/pdf"
                        )
                        
                        # Create combined PDF function
                        def create_combined_pdf(summary_content, risk_content):
                            pdf = FPDF()
                            
                            # Add summary page
                            pdf.add_page()
                            pdf.set_font("Arial", "B", 16)
                            pdf.cell(0, 10, "Legal Document Summary", ln=True, align="C")
                            pdf.ln(10)
                            
                            pdf.set_font("Arial", size=12)
                            summary_lines = summary_content.split('\n')
                            for line in summary_lines:
                                # Replace any non-latin1 characters
                                safe_line = ''.join(c if ord(c) < 256 else ' ' for c in line)
                                pdf.multi_cell(0, 10, txt=safe_line)
                            
                            # Add risk assessment page
                            pdf.add_page()
                            pdf.set_font("Arial", "B", 16)
                            pdf.cell(0, 10, "Risk Assessment Report", ln=True, align="C")
                            pdf.ln(10)
                            
                            pdf.set_font("Arial", size=12)
                            risk_lines = risk_content.split('\n')
                            for line in risk_lines:
                                # Replace any non-latin1 characters
                                safe_line = ''.join(c if ord(c) < 256 else ' ' for c in line)
                                pdf.multi_cell(0, 10, txt=safe_line)
                                
                            return pdf.output(dest="S").encode("latin-1")
                        
                        # Download combined report if summary exists
                        if st.session_state["final_summary"]:
                            combined_pdf_data = create_combined_pdf(st.session_state["final_summary"], final_risk_assessment)
                            st.download_button(
                                label="Download Complete Analysis",
                                data=combined_pdf_data,
                                file_name="legal_document_complete_analysis.pdf",
                                mime="application/pdf"
                            )
                    except Exception as e:
                        print("Error creating final risk assessment", e)
                        st.error(f"Error creating final risk assessment: {e}")
        
        # Show existing risk assessment if available
        elif "risk_assessment" in st.session_state and st.session_state["risk_assessment"]:
            st.subheader("Risk Assessment")
            st.write(st.session_state["risk_assessment"])
            
            # Create PDF function for existing risk assessment
            def create_risk_pdf(risk_content):
                pdf = FPDF()
                
                # Add risk assessment page
                pdf.add_page()
                pdf.set_font("Arial", "B", 16)
                pdf.cell(0, 10, "Risk Assessment Report", ln=True, align="C")
                pdf.ln(10)
                
                pdf.set_font("Arial", size=12)
                risk_lines = risk_content.split('\n')
                for line in risk_lines:
                    # Replace any non-latin1 characters
                    safe_line = ''.join(c if ord(c) < 256 else ' ' for c in line)
                    pdf.multi_cell(0, 10, txt=safe_line)
                    
                return pdf.output(dest="S").encode("latin-1")
            
            # Download button for existing risk assessment
            risk_pdf_data = create_risk_pdf(st.session_state["risk_assessment"])
            st.download_button(
                label="Download Risk Assessment",
                data=risk_pdf_data,
                file_name="legal_document_risk_assessment.pdf",
                mime="application/pdf"
            )
            
            # Combined PDF if summary exists
            if "final_summary" in st.session_state and st.session_state["final_summary"]:
                def create_combined_pdf(summary_content, risk_content):
                    pdf = FPDF()
                    
                    # Add summary page
                    pdf.add_page()
                    pdf.set_font("Arial", "B", 16)
                    pdf.cell(0, 10, "Legal Document Summary", ln=True, align="C")
                    pdf.ln(10)
                    
                    pdf.set_font("Arial", size=12)
                    summary_lines = summary_content.split('\n')
                    for line in summary_lines:
                        # Replace any non-latin1 characters
                        safe_line = ''.join(c if ord(c) < 256 else ' ' for c in line)
                        pdf.multi_cell(0, 10, txt=safe_line)
                    
                    # Add risk assessment page
                    pdf.add_page()
                    pdf.set_font("Arial", "B", 16)
                    pdf.cell(0, 10, "Risk Assessment Report", ln=True, align="C")
                    pdf.ln(10)
                    
                    pdf.set_font("Arial", size=12)
                    risk_lines = risk_content.split('\n')
                    for line in risk_lines:
                        # Replace any non-latin1 characters
                        safe_line = ''.join(c if ord(c) < 256 else ' ' for c in line)
                        pdf.multi_cell(0, 10, txt=safe_line)
                        
                    return pdf.output(dest="S").encode("latin-1")
                
                # Download combined report
                combined_pdf_data = create_combined_pdf(st.session_state["final_summary"], st.session_state["risk_assessment"])
                st.download_button(
                    label="Download Complete Analysis",
                    data=combined_pdf_data,
                    file_name="legal_document_complete_analysis.pdf",
                    mime="application/pdf"
                )
