import React, { ChangeEvent, useCallback, useEffect, useState } from "react";
import { urls } from "./urls";
import { Card, ICard } from "./Card";
import { clearIndex, crawlDocument } from "./utils";

import { Button } from "./Button";
interface ContextProps {
  className: string;
  // selected: string[] | null;
}

interface ResultPDF{
  chunks: number;
  message: string;
}

export const Context: React.FC<ContextProps> = ({ className }) => {
  const selected = null;
  const [entries, setEntries] = useState(urls);
  const [cards, setCards] = useState<ICard[]>([]);

  const [splittingMethod, setSplittingMethod] = useState("markdown");
  const [chunkSize, setChunkSize] = useState(256);
  const [overlap, setOverlap] = useState(1);
  const [url, setUrl] = useState('');
  const [urlRef, setUrlRef] = useState('');
  const [urlTitle, setUrlTitle] = useState('');
  const [tags, setTags] = useState('');
  const [category, setCategory] = useState('');

  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [resultPDF, setResultPDF] = useState<ResultPDF | null>(null);
  const [error, setError] = useState('');

  const DropdownLabel: React.FC<
    React.PropsWithChildren<{ htmlFor: string }>
  > = ({ htmlFor, children }) => (
    <label htmlFor={htmlFor} className="text-white p-2 font-bold">
      {children}
    </label>
  );

  const handleFileChange = async (e: any) => {
    setIsLoading(true)
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile);
      setError('');

      // send to API
      const formData = new FormData();
      formData.append('pdf', selectedFile);
      formData.append('splittingMethod', splittingMethod);
      formData.append('chunkSize', String(chunkSize));
      formData.append('overlap', String(overlap))
      

      const response = await fetch('/api/pdf', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Erro: ${response.status}`);
      }

      const data: ResultPDF = await response.json();
      setResultPDF(data);
      setIsLoading(false);
      setFile(null);

    } else {
      setFile(null);
      setIsLoading(false)
      setError('Por favor, selecione um arquivo PDF válido.');
    }
  };

  return (
    <div
      className={`flex flex-col border-2 overflow-y-auto rounded-lg border-gray-500 w-full ${className} bg-gray-300`}
    >
      <div className="flex flex-col items-start sticky top-0 w-full">
        <div className="flex flex-col items-start lg:flex-row w-full lg:flex-wrap p-2">
          {/* {buttons} */}
        </div>
        <div className="flex-grow w-full px-4">
        <input className="mb-2 w-full bg-with placeholder:text-gray-400 text-gray-600 text-sm border border-slate-200 rounded-md px-3 py-2 transition duration-300 ease focus:outline-none focus:border-slate-400 hover:border-slate-300 shadow-sm focus:shadow" 
            type="text"
            placeholder="Título da URL"
            value={urlTitle}
            onChange={(e: ChangeEvent<HTMLInputElement>) => setUrlTitle(e.target.value)} />
         
          <input className="mb-2 w-full bg-with placeholder:text-gray-400 text-gray-600 text-sm border border-slate-200 rounded-md px-3 py-2 transition duration-300 ease focus:outline-none focus:border-slate-400 hover:border-slate-300 shadow-sm focus:shadow"  
            type="text"
            placeholder="URL referência"
            value={urlRef}
            onChange={(e: ChangeEvent<HTMLInputElement>) => setUrlRef(e.target.value)} />

          <input className="mb-2 w-full bg-with placeholder:text-slate-400 text-slate-200 text-sm border border-slate-200 rounded-md px-3 py-2 transition duration-300 ease focus:outline-none focus:border-slate-400 hover:border-slate-300 shadow-sm focus:shadow"  
            type="text"
            placeholder="URL para ingestar"
            value={url}
            onChange={(e: ChangeEvent<HTMLInputElement>) => setUrl(e.target.value)} />

          <input className="mb-2  w-full bg-with placeholder:text-gray-400 text-gray-600 text-sm border border-slate-200 rounded-md px-3 py-2 transition duration-300 ease focus:outline-none focus:border-slate-400 hover:border-slate-300 shadow-sm focus:shadow"  
            type="text"
            placeholder="Tags separadas por virgula (Ex: tag1, tag2)"
            value={tags}
            onChange={(e: ChangeEvent<HTMLInputElement>) => setTags(e.target.value)} />

          {/* <input className="w-full bg-with placeholder:text-gray-400 text-gray-600 text-sm border border-slate-200 rounded-md px-3 py-2 transition duration-300 ease focus:outline-none focus:border-slate-400 hover:border-slate-300 shadow-sm focus:shadow"  
            type="text"
            placeholder="Categoria"
            value={category}
            onChange={(e: ChangeEvent<HTMLInputElement>) => setCategory(e.target.value)} /> */}

          <select className="w-full bg-with placeholder:text-gray-400 text-gray-600 text-sm border border-slate-200 rounded-md px-3 py-2 transition duration-300 ease focus:outline-none focus:border-slate-400 hover:border-slate-300 shadow-sm focus:shadow"
            value={category}
            onChange={(e: ChangeEvent<HTMLSelectElement>) => setCategory(e.target.value)}>
            <option value="">Selecione uma categoria</option>
            <option value="ecampus">e-Campus</option>
            <option value="sei">SEI</option>
            <option value="containstitucional">Conta Institucional</option>
            <option value="revista">Revista Eletrônica</option>
            <option value="assinador">Assina@UFVJM (assinatura)</option>
            <option value="eduroam">Eduroam (wifi)</option>
            <option value="dadosabertos">Dados Abertos</option>
            <option value="pagamentodigital">PAG@UFVJM (Pagamento Digital)</option>
            <option value="relatoriosgerenciais">Metabase (relatórios)</option>
            <option value="outros">Outros</option>
          </select>

          <Button
            className="w-full my-2 uppercase active:scale-[98%] transition-transform duration-100"
            style={{
              backgroundColor: "#4f6574",
              color: "white",
            }}
            onClick={() => crawlDocument(
                              {
                                url,
                                title: urlTitle,
                                tags,
                                category
                              },
                              setCards,
                              splittingMethod,
                              chunkSize,
                              overlap
                          )

            }
          >
            Adicionar URL
          </Button>
        </div>

        <div className="flex-grow w-full px-4">
          <h3 className="text-1xl font-bold mb-6">Indexar documento PDF</h3>

            <div className="mb-4">
              <label className="block mb-2">Selecione um arquivo PDF:</label>
              <input
                type="file"
                accept="application/pdf"
                onChange={handleFileChange}
                className="border p-2 w-full"
              />
            </div>

            {error && <p className="text-red-500 mb-4">{error}</p>}

            {isLoading && <p>Processando o PDF, por favor aguarde...</p>}
        </div>

        

        <div className="flex p-2"></div>
        <div className="text-left w-full flex flex-col rounded-b-lg bg-gray-600 p-3 subpixel-antialiased">
          <DropdownLabel htmlFor="splittingMethod">
            Método de Splitting:
          </DropdownLabel>
          <div className="relative w-full">
            <select
              id="splittingMethod"
              value={splittingMethod}
              className="p-2 bg-gray-700 rounded text-white w-full appearance-none hover:cursor-pointer"
              onChange={(e) => setSplittingMethod(e.target.value)}
            >
              <option value="recursive">Recursive Text Splitting</option>
              <option value="semantic">Semantic Text Splitting</option>
              <option value="markdown">Markdown Splitting</option>
            </select>
          </div>
          {splittingMethod === "recursive" && (
            <div className="my-4 flex flex-col">
              <div className="flex flex-col w-full">
                <DropdownLabel htmlFor="chunkSize">
                  Chunk Size: {chunkSize}
                </DropdownLabel>
                <input
                  className="p-2 bg-gray-700"
                  type="range"
                  id="chunkSize"
                  min={1}
                  max={2048}
                  onChange={(e) => setChunkSize(parseInt(e.target.value))}
                />
              </div>
              <div className="flex flex-col w-full">
                <DropdownLabel htmlFor="overlap">
                  Overlap: {overlap}
                </DropdownLabel>
                <input
                  className="p-2 bg-gray-700"
                  type="range"
                  id="overlap"
                  min={1}
                  max={200}
                  onChange={(e) => setOverlap(parseInt(e.target.value))}
                />
              </div>
            </div>
          )}
        </div>
      </div>
      <div className="flex flex-wrap w-full">
        {cards &&
          cards.map((card, key) => (
            <Card key={key} card={card} selected={selected} />
          ))}

          {resultPDF && (
              <div className="flex-grow w-full px-4">
                <h2 className="text-xl font-bold mb-2">Resultado da Extração</h2>
                <div className="border p-4 bg-gray-50 rounded">
                  <h3 className="font-bold mb-2">Número de chunks: {resultPDF.chunks}</h3>
                  <div className="max-h-96 overflow-y-auto">
                    {/* {resultPDF.chunks.map((chunk, index) => (
                      <div key={index} className="mb-4 p-3 border rounded bg-white">
                        <p className="text-sm font-bold mb-1">Chunk {index + 1}</p>
                        <p className="whitespace-pre-wrap">{chunk}</p>
                      </div>
                    ))} */}
                  </div>
                </div>
              </div>
            )}
      </div>

      <div className="flex-grow w-full px-4">
          <hr className="mb-4" />
          <Button
            className="w-full my-2 uppercase active:scale-[98%] transition-transform duration-100"
            style={{
              backgroundColor: "#4f6574",
              color: "white",
            }}
            onClick={() => clearIndex(setCards)}
          >
            Limpar Banco de Dados
          </Button>
        </div>


    </div>
  );
};
