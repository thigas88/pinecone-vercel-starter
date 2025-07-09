"use client"

import React, { useEffect, useRef, useState, FormEvent } from "react";
import Header from "@/app/components/Header";
import InstructionModal from "@/app/components/InstructionModal";
import { Context } from "@/app/components/Context";
import { AiFillGithub, AiOutlineInfoCircle } from "react-icons/ai";




const Page: React.FC = () => {
    const [isModalOpen, setModalOpen] = useState(false);

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
        <div className="flex w-full bg-muted flex-grow flex-wrap overflow-hidden ">

          <div className="duration-500 ease-in-out  w-full h-full overflow-y-auto md:static content-center lg:mx-2 rounded-lg">
            <Context className="" />
          </div>
          
        </div>
      </div>
    );
  };
  
  export default Page;