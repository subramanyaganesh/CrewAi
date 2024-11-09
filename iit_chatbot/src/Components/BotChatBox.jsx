import React from "react";
import { Bars } from 'react-loader-spinner'


export default function BotChatBox(props){
    console.log(props.message)
    return <div className="chatbot-bot-message">
        {props.message==="Loading" ? <Bars
            height="25"
            width="25"
            color="#D71313"
            ariaLabel="bars-loading"
            wrapperStyle={{}}
            wrapperClass=""
            visible={true}
            /> : props.message}
    </div>
}