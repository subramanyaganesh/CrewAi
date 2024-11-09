// in config.js
import { createChatBotMessage } from 'react-chatbot-kit';
import Avatar from '@mui/material/Avatar';
import BotChatBox from '../Components/BotChatBox';
import UserChatBox from '../Components/UserChatBox';


const botName = 'StanfordBot';

const config = {
  initialMessages: [createChatBotMessage(`Hello! I'm ${botName}, your personal assistant`)],
  botName: botName,
  customComponents: {
   header: () => <div className='chatbot-header'><Avatar alt='Hawk Bot' sx={{display:"inline-block",margin:"8px"}} src={require("../assets/stanfordLogo.jpg")} /> StanfordBot</div>,
   botAvatar: (props) => <Avatar alt='Hawk Bot'src={require("../assets/stanfordLogo.jpg")} {...props} />,
   botChatMessage: (props) => <BotChatBox {...props} />,
   userChatMessage: (props) => <UserChatBox {...props} />,
   userAvatar: (props) => null,
  },
  customStyles: {
    botMessageBox: {
      backgroundColor: '#376B7E',
    },
    chatButton: {
      backgroundColor: '#D71313',
    },
  },
};

export default config;