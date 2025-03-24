# LLaVA-Pool Data Documentation

This document provides detailed information about the data handling in LLaVA-Pool, focusing on the data formats, configuration, and usage examples.

## Data Formats

LLaVA-Pool supports various data formats for training and inference:

### Conversation Format

The conversation format is the fundamental structure for representing dialogues between users and assistants. It supports both text-based conversations and multimodal conversations with images.

### ShareGPT Format

ShareGPT format is the primary data format used in LLaVA-Pool. It structures conversations with clear role delineation and supports multimodal content.

The basic structure includes:
- `messages`: Array of conversation messages
- `images`: References to images included in the conversation

Each message contains:
- A role identifier (e.g., "user", "assistant")
- Content text

### Text-Only Data

Text-only data consists of conversations without any images or other media. This format is used for language model training and fine-tuning without visual components.

**Example:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is the capital of France?"
    },
    {
      "role": "assistant",
      "content": "The capital of France is Paris. It's known as the 'City of Light' and is famous for landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral."
    },
    {
      "role": "user",
      "content": "Can you tell me more about the Eiffel Tower?"
    },
    {
      "role": "assistant",
      "content": "The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris. It was named after engineer Gustave Eiffel, whose company designed and built the tower. Completed in 1889 as the entrance to the 1889 World's Fair, it stands 330 meters (1,083 ft) tall and was the tallest man-made structure in the world for 41 years until the completion of the Chrysler Building in New York in 1930. Today, it's one of the most recognizable landmarks in the world and receives around 7 million visitors annually."
    }
  ]
}
```

### Single-Image Data

Single-image data includes one image along with related text conversation. This format is commonly used for image captioning, visual question answering, and other single-image tasks.

**Example:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "<image>\nWhat can you tell me about this landmark?"
    },
    {
      "role": "assistant",
      "content": "This image shows the Eiffel Tower in Paris, France. It's one of the most iconic landmarks in the world. The tower is made of wrought iron and stands 330 meters (1,083 feet) tall. It was completed in 1889 for the World's Fair and was initially criticized by some of France's leading artists and intellectuals but has since become a global cultural icon of France and one of the most recognizable structures in the world."
    },
    {
      "role": "user",
      "content": "When is the best time to visit?"
    },
    {
      "role": "assistant",
      "content": "The best time to visit the Eiffel Tower is typically during the shoulder seasons of spring (April to June) or fall (September to October) when the weather is pleasant and there are fewer tourists compared to the summer peak season. Early mornings or evenings are recommended to avoid the longest lines. If you visit in the evening, you can see the tower illuminated with sparkling lights that turn on for five minutes every hour on the hour until 1 AM."
    }
  ],
  "images": ["path/to/eiffel_tower.jpg"]
}
```

### Multi-Image Data

Multi-image data includes multiple images within a conversation. This format is used for complex visual reasoning, image comparison, and tasks requiring analysis of multiple visual inputs.

**Example:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "<image>\nThis is the Eiffel Tower during the day."
    },
    {
      "role": "assistant",
      "content": "Yes, this is a daytime view of the Eiffel Tower in Paris. The image shows the full structure of the tower with its characteristic lattice design. The weather appears to be clear, giving a good view of this iconic landmark."
    },
    {
      "role": "user",
      "content": "<image>\nAnd this is how it looks at night. Which view do you prefer and why?"
    },
    {
      "role": "assistant",
      "content": "The nighttime view of the Eiffel Tower is quite spectacular! The tower is illuminated with golden lights that highlight its structure, and it appears there's also the famous sparkling light show happening (the 'Eiffel Tower Twinkle').\n\nBoth views have their unique appeal. The daytime view allows you to appreciate the intricate ironwork and engineering of the tower against the backdrop of the Paris skyline. The nighttime view transforms the tower into a magical, glowing beacon that dominates the Parisian night sky.\n\nPersonally, I find the nighttime view more enchanting because of the dramatic lighting that accentuates the tower's elegant design and creates a romantic atmosphere that Paris is famous for. However, both perspectives showcase different aspects of this magnificent structure's beauty."
    }
  ],
  "images": ["path/to/eiffel_tower_day.jpg", "path/to/eiffel_tower_night.jpg"]
}
```

### DPO Demo Data

The `dpo_demo.json` file contains demonstration data for the DPO (Demonstration Preference Optimization) process. This data is used to fine-tune models by providing examples of preferred and non-preferred responses to user inputs. Each entry in the file includes:

- `conversations`: An array of conversation messages between the user and the assistant.
- `chosen`: The preferred response from the assistant.
- `rejected`: The non-preferred response from the assistant.
- `images`: The path to the image associated with the conversation.

**Example:**
```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "<image> Describe the image."
      }
    ],
    "chosen": {
      "from": "gpt",
      "value": "The image shows a red telephone booth with a man standing inside."
    },
    "rejected": {
      "from": "gpt",
      "value": "There is a picture."
    },
    "images": "demo_data/COCO_train2014_000000222016.jpg"
  }
]
```

This structure helps in training models to understand and generate more accurate and contextually appropriate responses by learning from the preferred examples.

## Dataset Configuration

The `dataset_config.json` file defines the structure and format of datasets used in LLaVA-Pool. This configuration is crucial for properly loading and processing data.

### Configuration Structure

```json
{
  "dataset_name": {
    "file_name": "file.json",
    "formatting": "data_format",
    "columns": {
      "messages": "column_name",
      "images": "column_name"
    },
    "tags": {
      "role_tag": "tag_name",
      "content_tag": "tag_name",
      "user_tag": "user_value",
      "assistant_tag": "assistant_value"
    }
  }
}
```

### Key Components

The `src/llavapool/data` module contains several important components for data handling:

#### DatasetConfig Class

Defined in `dataset_config.py`, this class manages dataset configuration with properties:
- `dataset_name`: Name of the dataset
- `file_name`: JSON file containing the dataset
- `formatting`: Format of the data (e.g., "sharegpt")
- Additional configuration options like subset, split, and sample count

#### CommonColumns Class

Defines standard column names used across different data formats:
- `system`: Optional system messages
- `tools`: Optional tool content
- `images`: Optional image paths
- `videos`: Optional video paths
- `messages`: Conversation messages (default: "conversations")

#### ShareGPTTags Class

Defines tag names used in the ShareGPT format:
- `role_tag`: Tag for message role (default: "from")
- `content_tag`: Tag for message content (default: "value")
- `user_tag`: Value for user messages (default: "human")
- `assistant_tag`: Value for assistant messages (default: "gpt")
- Additional tags for system, function, and observation messages

## Using the Data Module
### Creating Custom Datasets

To create a custom dataset:

1. Prepare your data in one of the supported formats (preferably ShareGPT format)
2. Add your dataset configuration to `dataset_config.json`:
   ```json
   {
     "my_custom_dataset": {
       "file_name": "my_dataset.json",
       "formatting": "sharegpt",
       "columns": {
         "messages": "conversations",
         "images": "images"
       },
       "tags": {
         "role_tag": "role",
         "content_tag": "content",
         "user_tag": "user",
         "assistant_tag": "assistant"
       }
     }
   }
   ```
3. Load your dataset using the code example above

## Notes and Best Practices

- Image paths can be specified as relative or absolute paths
- Dataset files must be in JSON format
- For large datasets, consider using the `subset` and `num_samples` parameters to limit memory usage
- Ensure image files exist at the specified paths before training
- The `formatting` field must match one of the supported formats (e.g., "sharegpt")
- Custom tags can be defined for datasets with non-standard role and content field names
