import React from "react";
import Chatbot from 'react-chatbot-kit';
import 'react-chatbot-kit/build/main.css';
import config from "../dependencies/config";
import MessageParser from "../dependencies/MessageParser";
import ActionProvider from "../dependencies/ActionProvider";
import ChatIcon from '@mui/icons-material/Chat';
import ClearIcon from '@mui/icons-material/Clear';

export default function ChatBotBox(){
    const [showBot, toggleBot] = React.useState(false);

    function validate(text){
        if(text===""){
            return false
        }
        return true
    }

    const saveMessages = (messages, HTMLString) => {
        localStorage.setItem('chat_messages', JSON.stringify(messages));
      };
    
    const loadMessages = () => {
        const messages = JSON.parse(localStorage.getItem('chat_messages'));
        return messages;
    };

    return <><div className="chatbot-container">
        {showBot && <Chatbot
      config={config}
      messageParser={MessageParser}
      actionProvider={ActionProvider}
      saveMessages={saveMessages}
      loadMessages={loadMessages()}
    />}
    </div>
    <button className="chatbot-button" onClick={() => toggleBot((prev) => !prev)}>{showBot ? <ClearIcon fontSize="large"/> : <ChatIcon fontSize="large"/>}</button>
    </>
}