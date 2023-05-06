import { Tool } from "langchain/agents";
import { ChatOpenAI } from "langchain/chat_models";
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  MessagesPlaceholder,
  SystemMessagePromptTemplate,
} from "langchain/prompts";
import { BufferMemory } from "langchain/memory";
import { ConversationChain } from "langchain/chains";
import { Configuration } from "openai";
import { OpenAIApi } from "openai";

const openAIApiKey = process.env.OPENAI_API_KEY!;

const params = {
  verbose: true,
  temperature: 1,
  openAIApiKey,
  modelName: "gpt-3.5-turbo",
  maxConcurrency: 1,
  maxTokens: 1000,
  maxRetries: 5,
};

export class Model {
  public tools: Tool[] = [];
  public chain: ConversationChain;
  public openai: OpenAIApi;

  constructor() {
    const configuration = new Configuration({
      apiKey: openAIApiKey,
    });

    this.openai = new OpenAIApi(configuration);
    const model = new ChatOpenAI(params, configuration);

    const chatPrompt = ChatPromptTemplate.fromPromptMessages([
      SystemMessagePromptTemplate.fromTemplate(
        "You will act like a friendly person and engage in conversations and chats with users who are beginners of English. You are very talkative and witty, using simple vocabulary as much as possible for beginners to understand, and providing many specific details from their context. If you don't know the answer to the question, it will truthfully say it doesn't know."
      ),
      new MessagesPlaceholder("history"),
      HumanMessagePromptTemplate.fromTemplate("{input}"),
    ]);

    this.chain = new ConversationChain({
      memory: new BufferMemory({ returnMessages: true }),
      prompt: chatPrompt,
      llm: model,
    });
  }

  public async call(input: string) {
    const output = await this.chain.call({ input });
    return output.output;
  }
}
