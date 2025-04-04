import { Message } from "ai";
import { useRef, useState, useEffect } from "react";
import Markdown from "react-markdown";


export default function Messages({ messages, status }: { messages: Message[], status: string }) {
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const [showStatusMessage, setShowStatusMessage] = useState(false);

  useEffect(() => {
    if (status === 'submitted') {
      const timer = setTimeout(() => {
        setShowStatusMessage(true);
      }, 500);

      return () => clearTimeout(timer);
    } else {
      setShowStatusMessage(false);
    }
  }, [status]);

  return (
    <div className="border-2 border-gray-600 p-6 rounded-lg overflow-y-scroll flex-grow flex flex-col justify-end bg-gray-700">
      {messages.map((msg, index) => (
        <div
          key={index}
          className={`${
            msg.role === "assistant" ? "text-green-300" : "text-blue-300"
          } my-2 p-3 rounded shadow-md hover:shadow-lg transition-shadow duration-200 flex slide-in-bottom bg-gray-800 border border-gray-600 message-glow`}
        >
          <div className="rounded-tl-lg bg-gray-800 p-2 border-r border-gray-600 flex items-center">
            {msg.role === "assistant" ? "🤖" : "🧑‍💻"}
          </div>
          <div className="ml-2 message-item items-center text-gray-200 prose" >
            <Markdown>{msg.content}</Markdown>
          </div>
        </div>
      ))}
      {showStatusMessage && 
        <>
          <div
            className="text-green-300 my-2 p-3 rounded shadow-md hover:shadow-lg transition-shadow duration-200 flex slide-in-bottom bg-gray-800 border border-gray-600 message-glow"
          >
          <div className="rounded-tl-lg bg-gray-800 p-2 border-r border-gray-600 flex items-center">
            🤖
          </div>
          <div className="ml-2 message-item items-center text-gray-200 prose" >
            <p>Aguarde um momento, estou buscando informações....</p>
          </div>
        </div>
        </>
      }
      <div ref={messagesEndRef} />
    </div>
  );
}
