import React, { useRef, useState, useEffect } from "react";
import { Message } from "ai";
import { cn } from "@/lib/utils";
import {
  ChatBubble,
  ChatBubbleAction,
  ChatBubbleAvatar,
  ChatBubbleMessage,
} from "@/components/ui/chat/chat-bubble";
import { ChatMessageList } from "@/components/ui/chat/chat-message-list";
import CodeDisplayBlock from "@/components/code-display-block";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface ChatMessageComponentProps {
  message: Message;
  className?: string;
  status: string;
}

const ChatMessage: React.FC<ChatMessageComponentProps> = ({ message, className, status }) => {
  const isUserMessage = message.role === 'user';

  // const messagesEndRef = useRef<HTMLDivElement | null>(null);
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

    // <div className="flex h-full w-full max-w-3xl flex-col items-center mx-auto">
    //   <div className="flex-1 w-full overflow-y-auto py-6">
      
        <ChatBubble
            variant={message.role == "user" ? "sent" : "received"}
          >
            <ChatBubbleAvatar
              src=""
              fallback={message.role == "user" ? "👨🏽" : "🤖"}
            />
            <ChatBubbleMessage>
              {message.content
                .split("```")
                .map((part: string, index: number) => {
                  if (index % 2 === 0) {
                    return (
                      <Markdown key={index} remarkPlugins={[remarkGfm]}>
                        {part}
                      </Markdown>
                    );
                  } else {
                    return (
                      <pre className="whitespace-pre-wrap pt-2" key={index}>
                        <CodeDisplayBlock code={part} lang="" />
                      </pre>
                    );
                  }
                })}

            </ChatBubbleMessage>
          </ChatBubble>
    //   </div>
    // </div>

    
  );
};

export default ChatMessage;