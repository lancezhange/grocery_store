module.exports = {
  base: "/grocery_store/", // 设置站点根路径
  repo: "https://github.com/lancezhange/grocery_store",

  markdown: {
    extendMarkdown: md => {
      // var Plugin = require('markdown-it-regexp')

      // const termLinker = Plugin(/(\[TOC\])/, (match, utils) => {
      //   return `[[TOC]]`
      // })
      // md.use(termLinker)

      md.set({
        html: true
      });

      md.use(require("markdown-it-katex"));

      // iterator = require("markdown-it");
      // md.use(iterator, 'foo_replace', 'text', function (tokens, idx) {
      //   tokens[idx].content = tokens[idx].content.replace(/foo/g, 'bar');
      // });

      // md.use(require("markdown-it-table-of-contents"));
      // md.set({
      //   markerPattern: '/^\[toc\]/im'
      // });
    }
  },
  head: [
    [
      "link",
      {
        rel: "stylesheet",
        href: "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.7.1/katex.min.css"
      }
    ],
    [
      "link",
      {
        rel: "stylesheet",
        href: "https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/2.10.0/github-markdown.min.css"
      }
    ],
    [
      "link",
      {
        rel: "stylesheet",
        href: "/css/index.css" // 这个要注意，要在pulic文件中创建，否则无法打包，你可以通过css修改页面样式
      }
    ]
  ],
  plugins: [
    ["@vuepress/back-to-top"],
    ["@vuepress/medium-zoom"]
  ],

  title: "Lancezhange Vuepress Book",
  description: "Learner",
  themeConfig: {
    displayAllHeaders: true,
    activeHeaderLinks: false,
    sidebarDepth: 1,
    nav: [{
        text: "Home",
        link: "/"
      },
      {
        text: "Contents",
        link: "/SUMMARY/"
      },
      {
        text: "Blog",
        link: "http://www.lancezhange.com/"
      }
    ],
    // sidebar: "auto",
    sidebar: {
      "/section1/": ["" /* /foo/ */ , "data_structure", "basics", "basics2"],
      "/section2/": [
        "" /* /foo/ */ ,
        "osAndCompiler",
        "communicationAndProtocol",
        "re",
        "database",
        "designMode",
        "parallell",
        "system",
        "distributedSystem",
        "hpc"
      ],
      "/section3/": [
        "" /* /foo/ */ ,
        "cpp",
        "python",
        "java",
        "r",
        "scala",
        "javascript"
      ],
      "/section4/": [
        "" /* /foo/ */ ,
        "basicmath",
        "calculus",
        "algebra",
        "geometry",
        "probability",
        "graphTheory",
        "optimization",
        "statistics",
        "numerical_computation",
        "sampling",
        "mathProblems",
        "math_tips"
      ],
      "/section5/": ["" /* /foo/ */ , "game" /* /foo/one.html */ ],

      "/section6/": [
        "" /* /foo/ */ ,
        "ads",
        "ai-system",
        "ai",
        "ai_with_symbolism",
        "anomaly_detection",
        "attension",
        "backendArchitecture",
        "bayesian",
        "bigdata",
        "capsnet",
        "cluster",
        "cnn",
        "competition",
        "computerVision",
        "crf",
        "cryptocurrency",
        "ctr",
        "cuttingEdge",
        "dataAnalysis",
        "dataScrapy",
        "deepgbm",
        "deeplearning",
        "deeplearning_tricks",
        "em",
        "feature_enginerring",
        "few_shot_small_data_learning",
        "fm",
        "frontend",
        "gan",
        "gan2",
        "gbdt_xgboost",
        "gnn",
        "gradient_free",
        "hmm",
        "industry",
        "interview-questions",
        "knn",
        "knowledge_graph",
        "lda",
        "linearmodel",
        "lstm",
        "ltr",
        "mab",
        "mcmc",
        "mf",
        "ml_engineering",
        "mlconcepts",
        "mobile",
        "model_compress_simplify",
        "multimodel",
        "nlp",
        "nlp_reading_comprehension",
        "nlp_word2vec_embedding",
        "onlineLearning",
        "paper2018",
        "paper2019",
        "paper2020",
        "pca",
        "pgm",
        "qa",
        "rank",
        "rbm",
        "recommender",
        "recommender_practical_work",
        "rl",
        "rnn",
        "search_and_rank",
        "seq2seq",
        "socialnetworks",
        "spatialDataMining",
        "speech",
        "svm",
        "timeseries",
        "tricks",
        "vae",
        "vi-vae",
        "vi",
        "visualization",
        "wide_learning"
      ],

      "/section7/": [
        "" /* /foo/ */ ,
        "dailyTool",
        "dataMining",
        "blogResource",
        "dataResource"
      ],
      "/section8/": [
        "" /* /foo/ */ ,
        "block_chain",
        "books2014",
        "books2015",
        "books2016",
        "books2017",
        "books2018",
        "books2019",
        "books2020",
        "business",
        "career",
        "computer-culture",
        "entertainment",
        "hacker",
        "health",
        "mathculture",
        "movies2013",
        "movies2014",
        "movies2015",
        "movies2016",
        "movies2017",
        "movies2018",
        "movies2019",
        "movies2020",
        "opensource",
        "projectManagement",
        "selfmanagement",
        "travel2019",
        "travel2020"
      ],
      "/section9/": [
        "" /* /foo/ */ ,
        "buy_an_apartment",
        "financing",
        "fitness",
        "food",
        "japanese",
        "music",
        "painting",
        "paper-cut",
        "parenting",
        "photography",
        "pshchology"
      ],

      // fallback
      "/": [""]
    },
    // sidebar: [
    //   {
    //     title: "Home", // 必要的
    //     path: "/", // 可选的, 应该是一个绝对路径
    //     collapsable: false, // 可选的, 默认值是 true,
    //     sidebarDepth: 1, // 可选的, 默认值是 1
    //     children: ["/"]
    //   },
    //   {
    //     title: "Section1", // 必要的
    //     path: "/section1/", // 可选的, 应该是一个绝对路径
    //     collapsable: false, // 可选的, 默认值是 true,
    //     sidebarDepth: 3, // 可选的, 默认值是 1
    //     children: ["/section2/"]
    //   },
    //   {
    //     title: "Section2", // 必要的
    //     path: "/section2/", // 可选的, 应该是一个绝对路径
    //     collapsable: false, // 可选的, 默认值是 true,
    //     sidebarDepth: 3, // 可选的, 默认值是 1
    //     children: ["/section3/"]
    //   },
    //   {
    //     title: "Section3", // 必要的
    //     path: "/section3/", // 可选的, 应该是一个绝对路径
    //     collapsable: false, // 可选的, 默认值是 true,
    //     sidebarDepth: 3, // 可选的, 默认值是 1
    //     children: ["/section4/"]
    //   }
    // ],
    serviceWorker: {
      updatePopup: true // Boolean | Object, 默认值是 undefined.
      // 如果设置为 true, 默认的文本配置将是:
      // updatePopup: {
      //    message: "New content is available.",
      //    buttonText: "Refresh"
      // }
    }
  }
};