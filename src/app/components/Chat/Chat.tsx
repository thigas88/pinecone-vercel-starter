"use client";

import React, { FormEvent, ChangeEvent } from "react";
import { Button } from "@/components/ui/button";
import { ChatInput } from "@/components/ui/chat/chat-input";
import { useChat } from "ai/react";
import { Message } from "ai";
import { useEffect, useRef, useState } from "react";
import ChatMessage from './ChatMessage';
import { cn } from '@/lib/utils';
import { Loader2 } from 'lucide-react';
import {
  ChatBubble,
  ChatBubbleAction,
  ChatBubbleAvatar,
  ChatBubbleMessage,
} from "@/components/ui/chat/chat-bubble";
import { ChatMessageList } from "@/components/ui/chat/chat-message-list";
import {
  CopyIcon,
  CornerDownLeft,
  Mic,
  Paperclip,
  RefreshCcw,
  Send,
  Volume2,
} from "lucide-react";

const ChatAiIcons = [
  {
    icon: CopyIcon,
    label: "Copy",
  },
  {
    icon: RefreshCcw,
    label: "Refresh",
  },
  {
    icon: Volume2,
    label: "Volume",
  },
];

/**
 * Props for the Chat component.
 */
interface ChatProps {
  /**
   * The ID associated with the chat session.
   */
  id: string;
  input: string;
  handleInputChange: (e: ChangeEvent<HTMLInputElement>) => void;
  handleSubmit: (e: FormEvent<HTMLFormElement>) => Promise<void>;
  messages: Message[];
  status: string;
}

/**
 * The Chat component is a functional React component that provides a chat interface.
 * It uses the `useChat` hook from the `ai/react` library to manage chat messages.
 *
 * @param {ChatProps} props - The props passed to the component.
 * @returns {JSX.Element} The rendered Chat component.
 */
const Chat: React.FC<ChatProps> = ({ id }) => {
  const [isGenerating, setIsGenerating] = useState(false);
  /**
   * Destructuring the values returned from `useChat` hook.
   */
  const {
    messages,
    input,
    handleInputChange,
    handleSubmit,
    isLoading,
    status
  } = useChat({
    // api: `/api/chat?id=${id}`,
    initialMessages: [
      {
        role: "assistant",
        content: "Olá! Como posso te ajudar?",
        id: ''
      },
    ],
    onResponse(response) {
      if (response) {
        console.log(response);
        setIsGenerating(false);
      }
    },
    onError(error) {
      if (error) {
        setIsGenerating(false);
      }
    },
  });
  
  const onSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsGenerating(true);
    handleSubmit(e);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (isGenerating || isLoading || !input) return;
      setIsGenerating(true);
      onSubmit(e as unknown as React.FormEvent<HTMLFormElement>);
    }
  };

  const messagesWithoutSystem = messages.filter(
    (message) => message.role !== "system"
  );/**
  * Check if there are messages to display.
  */
  const hasMessages = messagesWithoutSystem.length > 0;
  /**
  * Check if the loading is active.
  */
  const hasLoading = isLoading;
    /**
  * Check if input should be disabled.
  */
  const disableInput = hasLoading;
  /**
  * Render the messages in the chat.
  */
  const renderMessages = () => {
    return (
      <div className="flex flex-col space-y-4">
        <div className="flex h-full w-full max-w-3xl flex-col items-center mx-auto">
          <div className="flex-1 w-full overflow-y-auto py-6"></div>
          <ChatMessageList>
            {messagesWithoutSystem.map((message, index) => (
              <ChatMessage
                key={index}
                message={message}
                status={status}
                className={cn({
                  "bg-primary text-primary-foreground": message.role === "user",
                })}
              />
            ))}
            {hasLoading && (
              <ChatBubble variant="received">
                <ChatBubbleAvatar src="" fallback="🤖" />
                <ChatBubbleMessage isLoading />
              </ChatBubble>
            )}
          </ChatMessageList>
        </div>
      </div>
    );
  };

  return (
    <div className="w-full max-w-2xl flex flex-col ">
      <div className='p-4 border-b'>
        <p className="font-bold"></p>
      </div>
      <div className="space-y-4 p-4 flex-1 overflow-y-auto">
        {hasMessages ? (
          renderMessages()
        ) : (
          <></>
        )}
      </div>
      <form 
        className="relative rounded-lg border bg-background focus-within:ring-1 focus-within:ring-ring" 
        onSubmit={handleSubmit} >
      <ChatInput
          disabled={disableInput}
          value={input}
          onKeyDown={onKeyDown}
          onChange={(e) => handleInputChange(e)}
          placeholder="Digite sua mensagem..."
          className="rounded-lg bg-background border-0 shadow-none focus-visible:ring-0"
        />
        <div className="flex items-center p-3 pt-0">
            <Button variant="ghost" size="icon">
              <Paperclip className="size-4" />
              <span className="sr-only">Anexar Arquivo</span>
            </Button>

            <Button variant="ghost" size="icon">
              <Mic className="size-4" />
              <span className="sr-only">Usar Microfone</span>
            </Button>

            <Button
              disabled={!input || isLoading}
              type="submit"
              size="sm"
              className="ml-auto gap-1.5"
            >
              Enviar Mensagem
              <CornerDownLeft className="size-3.5" />
            </Button>
          </div>
      </form>
    </div>
  );
};

export default Chat;