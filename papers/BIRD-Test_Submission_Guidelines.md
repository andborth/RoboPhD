**BIRD-Test Submission Guidelines:**

Thank you for your interest in our work. Please send your request along with the essential files or descriptions to [bird.bench23@gmail.com](mailto:bird.bench23@gmail.com). Currently, we support four types of evaluations:  
*To protect our data, each code submission will be examined seriously, please not include any private API package which can upload our database. Therefore, please make your submission files concise only containing related files about your work, please remove irrelevant files. This could lead to faster and smoother evaluation. Thanks\!*

**1\. Single A100 80G GPU Inference (0\~34B, suitable for smaller models):**  
**Expected Time: 1-10 days (highly depend on your env and instruction file)**  
    • Provide a detailed Readme file and compressed code. Please make sure your code is      
      successful on your dev evaluation.**(Required)**  
     • Provide a **requirement.txt** for your environment package up. For special env, like java,    
      jdk, please illustrate them in readme. Our cuda version is 12.2 or 12.3.  
    • Push your model to [huggingface](https://huggingface.co/models) with appropriate privacy. You can refer to these docs:   
      [doc1](https://juejin.cn/post/7081452948550746148), [doc2](https://huggingface.co/docs/transformers/v4.15.0/model_sharing#:~:text=In%20order%20to%20upload%20a,can%20use%20the%20transformers%2Dcli%20.). (Highly Suggested)

**2\. Multi-GPU Parallel Evaluation (\> 34B, suitable for open-source LLMs):**  
**Expected Time: 10 days (highly depend on your env and instruction file)**  
    • Provide a clear Readme file and compressed code. Please make sure your code is   
      successful on your dev evaluation. **(Required)**  
    • Provide a **requirement.txt** for your environment package up. For special env, like java,    
      jdk, please illustrate them in readme. Our cuda version is 12.2 or 12.3.  
    • Push your model to [huggingface](https://huggingface.co/models) with appropriate privacy. You can refer to these docs:   
      [doc1](https://juejin.cn/post/7081452948550746148), [doc2](https://huggingface.co/docs/transformers/v4.15.0/model_sharing#:~:text=In%20order%20to%20upload%20a,can%20use%20the%20transformers%2Dcli%20.). (Highly Suggested)  
    • Specify resources required for model inference on your dev evaluation (e.g., 4 A100 80G   
      cards, 3 hours) for our appropriate resource allocation. **(Required)**

**3\. Evaluation via API Call (Suitable for closed-source LLMs):**  
**Expected Time: 1-5 days**  
    • Provide a clear Readme file and compressed code. Please make sure your code is   
      successful on your dev evaluation. **(Required)**  
    • Please provide your own keys if your LLMs need APIs out of the above **(Required)**  
      keys. And you could reset the keys after the evaluation terminates **(Required)**  
    • Inform us of the number of prompt tokens on your dev environment in advance for us to   
      estimate costs on testing. **(Required)**

**4\. Combined Models (Suitable for Tool-based LLMs or Closed / Open LLMs Mixed):**  
**Expected Time: 10-20 days (highly depend on your env and instruction file)**  
    • Provide a clear Readme file and compressed code. Please make sure your code is   
      successful on your dev evaluation. **(Required)**  
    • Please provide your own keys if your LLMs need APIs out of the above   
      keys. And you could reset the keys after the evaluation terminates**(Required)**  
    • Inform us of the number of prompt tokens on your dev environment in advance for us to   
      estimate costs on testing. **(Required)**  
    • Provide a **requirement.txt** for your environment package up. For special env, like java,    
      jdk, please illustrate them in readme. Our cuda version is 12.2 or 12.3.  
    • Push your models to the modelscope please if your models are large and want us to   
      return results soon.  
    • Also, it would be very helpful if you could separate between GPU-based codes and   
      Closed LLM codes.  
    • Push your model to [huggingface](https://huggingface.co/models) with appropriate privacy. You can refer to these docs:   
      [doc1](https://juejin.cn/post/7081452948550746148), [doc2](https://huggingface.co/docs/transformers/v4.15.0/model_sharing#:~:text=In%20order%20to%20upload%20a,can%20use%20the%20transformers%2Dcli%20.). (Highly Suggested)  
    • Specify resources required for model inference on your dev evaluation (e.g., 4 A100 80G   
      cards, 3 hours) for our appropriate resource allocation. **(Required)**

**Test Set Input:**  
**test\_databases:** The same with dev databases, but with some giant databases.  
**test\_tables.json:** The same with dev\_tables.json  
column\_meaning.json: New, the summarized database description files, the same with [https://github.com/quge2023/TA-SQL/blob/master/outputs/column\_meaning.json](https://github.com/quge2023/TA-SQL/blob/master/outputs/column_meaning.json).  
**test.json:** The same with dev.json, but doesn’t contain “SQL”

**Submission Required Material:**  
**Readme.md:** Please include detailed submission instructions with relevant commands to save significant time.  
**Code Zip:** Provide your code in a compressed zip file.  
**Models or Keys:** Upload your models to Hugging Face or ModelScope (recommended). Or OpenAI or other API keys.   
**Columeaning File Usage:** Please also state whether you need \`column\_meaning.json\` in testing.  
**Dev SQL File:** To facilitate following and reproducing your results, please include your predicted SQLs on the development set. However, if you have any concerns of not open-sourcing your dev results, please also let us know.

**Submission Frequency:**   
Please note that we allow each team to provide up to **2** checkpoints per submission and **1-2** submission within a **2-month** period. Also you can only choose at most 2 preferred results to update at each submission.

**NOTE:**  
1\. Kindly be aware that in real-world scenarios, the ground truth SQL queries are **NOT** observed. Therefore, please ensure that your method should not rely on these ground truth SQLs. Otherwise, this may lead to failure of your code. Each test case in test.json is structured as follows (it’s just an example not the exact data from test, and SQL is an **EMPTY** string):  
{  
        "db\_id": “nba\_data”,  
        "question": "how many players in NBA?",  
        "evidence": "’how many’ refers to COUNT()",  
        "SQL": ""  
    }

2.Please ensure your code includes **logging** or an **error-handling mechanism**. By this way, after debugging, we can restart the evaluation from the example where errors occurred, rather than starting over. This will save both time and money. Logging files will make us better communicate when bugs happen.   
*(If your codes require starting over if bugs happen, we will not be responsible for token costs. Thanks for your understanding\!)*

**Disclaimer**: We will only use your code for evaluation purposes and will not disseminate or disclose any details of your code. After the evaluation is completed and confirmed with the model author, we will immediately delete the server instance, including your code and Docker.