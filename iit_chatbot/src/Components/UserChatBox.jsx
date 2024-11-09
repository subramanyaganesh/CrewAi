import React from "react";


export default function UserChatBox(props){
    return <div className="chatbot-user-message">
        {props.message}
    </div>
}