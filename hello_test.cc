#include <gtest/gtest.h>
#include "../bloom/bloom.h"
#include "../../bloom_tokenizer/tokenization.h"
#include <fstream>
#include <iomanip>
#include <type_traits>
#include <filesystem>
#include <cmath>
#define PI 3.14159265358979323846

std::vector<std::string> checkpoints;


extern std::string input_start;
extern std::string input_end;
extern std::regex pat;

extern std::unordered_map<std::string, std::vector<std::string>> cache;
extern std::unordered_map<std::string, int> encoder;
extern std::unordered_map<int, std::string> decoder;
extern std::vector<std::tuple<std::string, std::string>> bpe_merges;
extern std::unordered_map<std::string, int> byte_decoder;
extern std::unordered_map<int, std::string> byte_encoder;
extern std::unordered_map<std::tuple<std::string, std::string>, float, hash_tuple> bpe_ranks;


float trainTest()
{
    BLOOMConfig config;


    std::vector<std::pair<std::string, std::string>> dataset;

    std::string inputs = "假设你是一位Airbnb房主。\n请帮助我检查我的Airbnb列表，找出任何拼写错误，排版错误或价格错误。\n";
    std::string output = "很抱歉，作为AI语言模型，我无法检查您的Airbnb列表。建议您仔细检查您的列表，确保所有信息都准确无误。您也可以请其他人帮助您检查列表，以确保它们符合Airbnb的标准。";


    std::string str = input_start + inputs + input_end + output + "</s>";
    std::cout << "str: " << str << std::endl;
    std::tuple<std::vector<int>, std::vector<int>> tokenize = tokenizer(str, "/home/yinxun/phoenix-inst-chat-7b-DLC/bloom_tokenizer/tokenizer.json");
    std::tuple<std::vector<int>, std::vector<int>> tokenizeDup = tokenizer("<s>" + output + "</s>", "/home/yinxun/phoenix-inst-chat-7b-DLC/bloom_tokenizer/tokenizer.json");
    vec1d_t input(std::get<0>(tokenize).begin(), std::get<0>(tokenize).end());
    vec1d_t_i attention_mask = std::get<1>(tokenize);


    vec1d_t_i label = std::get<0>(tokenizeDup);
    vec1d_t_i feeble(input.size() - label.size(), -100);
    label.insert(label.begin(), feeble.begin(), feeble.end());
        

    for (int i = 0; i < 1; ++i)
    {
        std::cout << "********** arg: " << i << " **********" << std::endl;
        std::cout << "input: " << std::endl;
        vec1d_t feed = input;
        std::cout << "feed: " << feed.size() << std::endl;
        for (int j = 0; j < feed.size(); j++)
        {
            std::cout << feed[j] << " ";
            if ((j + 1) % 9 == 0) std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout << "label: " << std::endl;
        for (int j = 0; j < label.size(); j++)
        {
            std::cout << label[j] << " ";
            if ((j + 1) % 9 == 0) std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    bool use_cache = true;
    bool output_attentions = false;
    bool output_hidden_states = false;
    bool return_dict = true;

    vec5d_t_i head_mask = {};
    vec3d_t input_embeds = {};
    std::vector<std::tuple<vec3d_t, vec3d_t>> past_key_values;
    
    std::tuple<std::map<std::string, std::any>, std::map<std::string, std::any>> Pair = get_dataloader(config);

    std::map<std::string, std::any> WeightSeries = std::get<0>(Pair);
    std::map<std::string, std::any> InputSeries = std::get<1>(Pair);


    std::tuple<float, vec3d_t, std::vector<std::tuple<vec3d_t, vec3d_t>>, std::vector<vec3d_t>, std::vector<vec4d_t>> 
    outputs = BloomForCausalLM(
                config, 
                vec2d_t(1, input), 
                past_key_values, 
                vec2d_t_i(1, attention_mask),
                head_mask,
                input_embeds,
                vec2d_t_i(1, label),
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
                "transformer",
                WeightSeries,
                InputSeries
    );
    float loss = std::get<0>(outputs);
    std::cout << "loss: " << loss << std::endl;

    return loss;
}

// 测试正数
TEST(FactorialTest, Positive)
{
    EXPECT_EQ(1.0, std::abs(-1));
    // ASSERT_NEAR(2.84882, trainTest(), 0.01);
}