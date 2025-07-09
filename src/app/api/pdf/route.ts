import { NextResponse } from 'next/server';
import { WebPDFLoader } from "@langchain/community/document_loaders/web/pdf";

// import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import { getEmbeddings } from "@/app/utils/embeddings";
import { Document, MarkdownTextSplitter, RecursiveCharacterTextSplitter } from "@pinecone-database/doc-splitter";
import { Pinecone, PineconeRecord, ServerlessSpecCloudEnum } from "@pinecone-database/pinecone";
import { chunkedUpsert } from '../../utils/chunkedUpsert'
import { truncateStringByBytes } from "@/app/utils/truncateString"
// @ts-ignore
import { Corpus } from "tiny-tfidf";// https://github.com/kerryrodden/tiny-tfidf
import md5 from "md5";
import { pineconeSetup } from "@/app/components/Context/utils";



type DocumentSplitter = RecursiveCharacterTextSplitter | MarkdownTextSplitter


// Interface para representar palavras-chave e suas pontuações
interface Keyword {
  word: string;
  score: number;
}


// Lista de stopwords em português do Brasil
const stopwords: string[] = [
  "a", "à", "ao", "aos", "as", "às", "com", "como", "da", "das", "de", "dela", "dele", "deles", "delas", "do", "dos",
  "e", "é", "em", "entre", "era", "essa", "esse", "esta", "está", "estar", "estão", "eu", "foi", "for", "foram",
  "havia", "isso", "isto", "já", "lá", "mas", "me", "mesmo", "muito", "na", "não", "nas", "nem", "no", "nos", "nós",
  "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "porque", "quando", "que", "quem", "se", "sem",
  "ser", "seu", "sua", "são", "também", "tem", "tendo", "ter", "teu", "teve", "tinha", "tive", "tu", "um", "uma", "você",
  "vocês", "já", "dele", "nela", "nele", "destes", "daqueles", "aquelas", "aquilo"
];


export async function POST(req: Request) {

  
  // Initialize the Pinecone client
  const pinecone = new Pinecone();

  try {


    // Obter o FormData da requisição
    const formData = await req.formData();
    const pdfFile = formData.get('pdf');
    const splittingMethod = formData.get('splittingMethod');
    const chunkSize = Number(formData.get('chunkSize'));
    const chunkOverlap = Number(formData.get('overlap'));
;    
    if (!pdfFile || !(pdfFile instanceof File)) {
      return NextResponse.json(
        { error: 'Nenhum arquivo PDF enviado' },
        { status: 400 }
      );
    }
    
    // Verificar se é um PDF
    if (pdfFile.type !== 'application/pdf') {
      return NextResponse.json(
        { error: 'O arquivo enviado não é um PDF válido' },
        { status: 400 }
      );
    }

    // Converter o arquivo para um Blob
    const fileBlob = new Blob([await pdfFile.arrayBuffer()], { type: 'application/pdf' });
    
    // Usar LangChain para carregar e processar o PDF
    const loader = new WebPDFLoader(fileBlob, {
      splitPages: false // Mantém o documento como um único documento
    });
    
    const docs = await loader.load();
    
    // Verificar se conseguimos extrair algum conteúdo
    if (!docs || docs.length === 0) {
      return NextResponse.json(
        { error: 'Não foi possível extrair texto do PDF' },
        { status: 400 }
      );
    }

    // // Criar o text splitter para chunking
    // const textSplitter = new RecursiveCharacterTextSplitter({
    //   chunkSize: 1000,
    //   chunkOverlap: 200,
    // });

    // Choose the appropriate document splitter based on the splitting method
    const splitter: DocumentSplitter = splittingMethod === 'recursive' ?
      new RecursiveCharacterTextSplitter({ chunkSize, chunkOverlap }) : new MarkdownTextSplitter({});

    // Prepare documents by splitting the pages
    const documents = await prepareDocument(docs[0], splitter);


    // Create Pinecone index if it does not exist
    const indexList: string[] = (await pinecone.listIndexes())?.indexes?.map(index => index.name) || [];
    const indexExists = indexList.includes(pineconeSetup.indexName);
    if (!indexExists) {
      await pinecone.createIndex({
        name: pineconeSetup.indexName,
        dimension: 768,
        waitUntilReady: true,
        spec: { 
          serverless: { 
              cloud: pineconeSetup.cloudName, 
              region: pineconeSetup.regionName
          }
        } 
      });
    }

    const index = pinecone.Index(pineconeSetup.indexName)

    // Get the vector embeddings for the documents
    const vectors = await Promise.all(documents.flat().map(embedDocument));

    // Upsert vectors into the Pinecone index
    await chunkedUpsert(index!, vectors, '', 10);

    // Dividir o texto em chunks
    // const chunks = await textSplitter.splitText(docs[0].pageContent);

    // console.log('Foi gerado ' + chunks.length + ' chunks');

    // Retornar os chunks
    return NextResponse.json({ 
      message: 'PDF processado com sucesso',
      chunks: vectors.length
    });

    // const documents = await seed(
    //   pdf,
    //   1,
    //   process.env.PINECONE_INDEX!,
    //   process.env.PINECONE_CLOUD as ServerlessSpecCloudEnum || 'aws',
    //   process.env.PINECONE_REGION || 'us-west-2',
    //   options
    // )
    // return NextResponse.json({ success: true, documents })
  } catch (error) {
    console.error('Erro ao processar o PDF:', error);
    return NextResponse.json({ success: false, error: "Failed extract pdf content" })
  }

}



async function embedDocument(doc: Document): Promise<PineconeRecord> {
  try {
    // Generate OpenAI embeddings for the document content
    const embedding = await getEmbeddings(doc.pageContent);

    // Create a hash of the document content
    const hash = md5(doc.pageContent);

    // Return the vector embedding object
    return {
      id: hash, // The ID of the vector is the hash of the document content
      values: embedding, // The vector values are the OpenAI embeddings
      metadata: { // The metadata includes details about the document
        chunk: doc.pageContent, // The chunk of text that the vector represents
        text: doc.metadata.text as string, // The text of the document
        //url: doc.metadata.url as string, // The URL where the document was found
        hash: doc.metadata.hash as string, // The hash of the document content
        keywords: doc.metadata.keywords as [], // The keywords associated with the document
      }
    } as PineconeRecord;
  } catch (error) {
    console.log("Error embedding document: ", error)
    throw error
  }
}


async function prepareDocument(page: Document, splitter: DocumentSplitter): Promise<Document[]> {
  // Get the content of the page
  const pageContent = page.pageContent;

  // Split the documents using the provided splitter
  const docs = await splitter.splitDocuments([
    new Document({
      pageContent,
      metadata: {
        //url: page.url,
        // Truncate the text to a maximum byte length
        text: truncateStringByBytes(pageContent, 36000)
      },
    }),
  ]);

  // Map over the documents and add a hash to their metadata
  const processedDocs =  docs.map((doc: Document) => {

    const docKeywords: Keyword[] = []
    // Extract keywords from doc.pageContent using langchain
    const textId = md5(doc.pageContent)
    const corpus = new Corpus(
      [textId],
      [doc.pageContent],
      false,
      stopwords
    )

    const corpusTerms: [] = corpus.getTopTermsForDocument(textId);

    corpusTerms.forEach((term) => {
      docKeywords.push({
        word: (term[0] as unknown as string) || '',
        score: (term[1] as unknown as number) || 0
      })
    })

    console.log("Palavras-chave do chunk "+ md5(doc.pageContent) +":", docKeywords as Keyword[]);

    return {
      pageContent: doc.pageContent,
      metadata: {
        ...doc.metadata,
        // Create a hash of the document content
        hash: md5(doc.pageContent),
        keywords: JSON.stringify(docKeywords)
      },
    };
  });

  return processedDocs;
}

