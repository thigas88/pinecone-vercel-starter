import { IUrlEntry } from "./UrlButton";
import { ICard } from "./Card";
import { ServerlessSpecCloudEnum } from '@pinecone-database/pinecone'


export const pineconeSetup = {
  indexName: process.env.PINECONE_INDEX!,
  pineconeApiKey: process.env.PINECONE_API_KEY!,
  pineconeEnvironment: process.env.PINECONE_ENVIRONMENT!,
  cloudName: process.env.PINECONE_CLOUD as ServerlessSpecCloudEnum || 'aws', 
  regionName: process.env.PINECONE_REGION || 'us-west-2',
}

export async function crawlDocument(
  url: string,
  setEntries: React.Dispatch<React.SetStateAction<IUrlEntry[]>>,
  setCards: React.Dispatch<React.SetStateAction<ICard[]>>,
  splittingMethod: string,
  chunkSize: number,
  overlap: number
): Promise<void> {
  setEntries((seeded: IUrlEntry[]) =>
    seeded.map((seed: IUrlEntry) =>
      seed.url === url ? { ...seed, loading: true } : seed
    )
  );
  const response = await fetch("/api/crawl", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      url,
      options: {
        splittingMethod,
        chunkSize,
        overlap,
      },
    }),
  });

  const { documents } = await response.json();

  setCards(documents);

  setEntries((prevEntries: IUrlEntry[]) =>
    prevEntries.map((entry: IUrlEntry) =>
      entry.url === url ? { ...entry, seeded: true, loading: false } : entry
    )
  );
}

export async function clearIndex(
  setEntries: React.Dispatch<React.SetStateAction<IUrlEntry[]>>,
  setCards: React.Dispatch<React.SetStateAction<ICard[]>>
) {
  const response = await fetch("/api/clearIndex", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });

  if (response.ok) {
    setEntries((prevEntries: IUrlEntry[]) =>
      prevEntries.map((entry: IUrlEntry) => ({
        ...entry,
        seeded: false,
        loading: false,
      }))
    );
    setCards([]);
  }
}