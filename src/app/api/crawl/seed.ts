import { getEmbeddings } from "@/utils/embeddings";
import { Document, MarkdownTextSplitter, RecursiveCharacterTextSplitter } from "@pinecone-database/doc-splitter";
import { Pinecone, PineconeRecord, ServerlessSpecCloudEnum } from "@pinecone-database/pinecone";
import { chunkedUpsert } from '../../utils/chunkedUpsert'
import md5 from "md5";
import { Crawler, Page } from "./crawler";
import { truncateStringByBytes } from "@/utils/truncateString"
import { Corpus } from "tiny-tfidf";
// https://github.com/kerryrodden/tiny-tfidf



interface SeedOptions {
  splittingMethod: string
  chunkSize: number
  chunkOverlap: number
}

// Interface para representar palavras-chave e suas pontuações
interface Keyword {
  word: string;
  score: number;
}

type DocumentSplitter = RecursiveCharacterTextSplitter | MarkdownTextSplitter


// Lista de stopwords em português do Brasil
const stopwords: string[] = [
  "a", "à", "ao", "aos", "as", "às", "com", "como", "da", "das", "de", "dela", "dele", "deles", "delas", "do", "dos",
  "e", "é", "em", "entre", "era", "essa", "esse", "esta", "está", "estar", "estão", "eu", "foi", "for", "foram",
  "havia", "isso", "isto", "já", "lá", "mas", "me", "mesmo", "muito", "na", "não", "nas", "nem", "no", "nos", "nós",
  "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "porque", "quando", "que", "quem", "se", "sem",
  "ser", "seu", "sua", "são", "também", "tem", "tendo", "ter", "teu", "teve", "tinha", "tive", "tu", "um", "uma", "você",
  "vocês", "já", "dele", "nela", "nele", "destes", "daqueles", "aquelas", "aquilo"
];


async function seed(url: string, limit: number, indexName: string, cloudName: ServerlessSpecCloudEnum, regionName: string, options: SeedOptions) {
  try {
    // Initialize the Pinecone client
    const pinecone = new Pinecone();

    // Destructure the options object
    const { splittingMethod, chunkSize, chunkOverlap } = options;

    // Create a new Crawler with depth 1 and maximum pages as limit
    const crawler = new Crawler(5, limit || 100);

    // Crawl the given URL and get the pages
    const pages = await crawler.crawl(url) as Page[];

    // @todo If splittingMethod === 'semantic' need implement a semantic splitter in the future in python  service

    // Choose the appropriate document splitter based on the splitting method
    const splitter: DocumentSplitter = splittingMethod === 'recursive' ?
      new RecursiveCharacterTextSplitter({ chunkSize, chunkOverlap }) : new MarkdownTextSplitter({});

    // Prepare documents by splitting the pages
    const documents = await Promise.all(pages.map(page => prepareDocument(page, splitter)));

    // Create Pinecone index if it does not exist
    const indexList: string[] = (await pinecone.listIndexes())?.indexes?.map(index => index.name) || [];
    const indexExists = indexList.includes(indexName);
    if (!indexExists) {
      await pinecone.createIndex({
        name: indexName,
        dimension: 768,
        waitUntilReady: true,
        spec: { 
          serverless: { 
              cloud: cloudName, 
              region: regionName
          }
        } 
      });
    }

    const index = pinecone.Index(indexName)

    // Get the vector embeddings for the documents
    const vectors = await Promise.all(documents.flat().map(embedDocument));

    // Upsert vectors into the Pinecone index
    await chunkedUpsert(index!, vectors, '', 10);

    // Return the first document
    return documents[0];
  } catch (error) {
    console.error("Error seeding:", error);
    throw error;
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
        url: doc.metadata.url as string, // The URL where the document was found
        hash: doc.metadata.hash as string, // The hash of the document content
        keywords: doc.metadata.keywords as [], // The keywords associated with the document
      }
    } as PineconeRecord;
  } catch (error) {
    console.log("Error embedding document: ", error)
    throw error
  }
}

async function prepareDocument(page: Page, splitter: DocumentSplitter): Promise<Document[]> {
  // Get the content of the page
  const pageContent = page.content;

  // Split the documents using the provided splitter
  const docs = await splitter.splitDocuments([
    new Document({
      pageContent,
      metadata: {
        url: page.url,
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


export default seed;