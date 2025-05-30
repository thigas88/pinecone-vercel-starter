"use client";

import React, { useEffect, useRef, useState, FormEvent } from "react";
import { Context } from "@/app/components/Context";
import Header from "@/app/components/Header";
import Chat from "@/app/components/Chat/Chat";
import { useChat } from "ai/react";
import InstructionModal from "@/app/components/InstructionModal";
import { AiFillGithub, AiOutlineInfoCircle } from "react-icons/ai";

const Page: React.FC = () => {
  const [gotMessages, setGotMessages] = useState(false);
  // const [context, setContext] = useState<string[] | null>(null);
  const [isModalOpen, setModalOpen] = useState(false);

  const { messages, input, handleInputChange, handleSubmit, status } = useChat({
    onFinish: async (message) => {
      setGotMessages(true);
      console.log('Finished streaming message:', message);
    },
    onError: error => {
      console.error('An error occurred:', error);
    },
    onResponse: response => {
      console.log('Received HTTP response from server:', response);
    },
  });

  const prevMessagesLengthRef = useRef(messages.length);

  const handleMessageSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    handleSubmit(e);
    // setContext(null);
    setGotMessages(false);
  };

  // useEffect(() => {
  //   const getContext = async () => {
  //     const response = await fetch("/api/context", {
  //       method: "POST",
  //       body: JSON.stringify({
  //         messages,
  //       }),
  //     });
  //     const { context } = await response.json();
  //     setContext(context.map((c: any) => c.id));
  //   };
  //   if (gotMessages && messages.length >= prevMessagesLengthRef.current) {
  //     getContext();
  //   }

  //   prevMessagesLengthRef.current = messages.length;
  // }, [messages, gotMessages]);

  return (
    <div className="flex flex-col justify-between h-screen p-2 mx-auto max-w-full">
      <Header className="my-5" />

      <button
        onClick={() => setModalOpen(true)}
        className="fixed right-4 top-4 md:right-6 md:top-6 text-xl text-white animate-pulse-once info-button"
      >
        <AiOutlineInfoCircle />
      </button>

      <InstructionModal
        isOpen={isModalOpen}
        onClose={() => setModalOpen(false)}
      />
      <div className="flex w-full bg-muted flex-grow overflow-hidden relative">
        <Chat 
          messages={messages} 
          status={status} 
          handleSubmit={handleMessageSubmit} 
          handleInputChange={handleInputChange} 
          input={input}
          id="" />
        <div className="absolute transform translate-x-full transition-transform duration-500 ease-in-out right-0 w-2/3 h-full overflow-y-auto lg:static lg:translate-x-0 lg:w-2/5 lg:mx-2 rounded-lg">
          {/* <Context className="" /> */}
        </div>
      </div>
    </div>
  );
};

export default Page;
