export const paper = {
  title: "Agentic ESOpt: Fine-Tuning Long-Horizon LLM Agents with Minimal GPU Memory Requirements",
  authors: [
    { name: "Zhi Zheng", affiliation: 1 },
    { name: "Rongsheng Chen", affiliation: 2 },
    { name: "Yunpeng Ba", affiliation: 2 },
    { name: "Zhenkun Wang", affiliation: 2 },
    { name: "Yee Whye Teh", affiliation: 3 },
    { name: "Wee Sun Lee", affiliation: 1 },
  ],
  affiliations: [
    { id: 1, name: "National University of Singapore" },
    { id: 2, name: "Southern University of Science and Technology" },
    { id: 3, name: "Oxford" },
  ],
  correspondingEmail: "zhi.zheng@u.nus.edu",
  githubUrl: "https://github.com/zz1358m/Agentic-ESOpt",
  checkpointCollectionUrl: "https://huggingface.co/collections/zz1358m/agentic-esopt-checkpoints-collection-6a781bb727d86d7742e61df6",
  dailyPapersUrl: "https://huggingface.co/papers",
  acknowledgement: "We would like to sincerely thank Jiaying Wu, Penghui Qi, Zichen Liu, and Ziqiao Meng from the National University of Singapore, as well as Zi'ang Li from Human& for their important comments on the methodology and paper-writing.",
  year: 2026,
} as const;

export const paperCitation = `@article{agentic-esopt2026,
  title = {${paper.title}},
  author = {Zheng, Zhi and Chen, Rongsheng and Ba, Yunpeng and Wang, Zhenkun and Teh, Yee Whye and Lee, Wee Sun},
  year = {${paper.year}},
  url = {${paper.githubUrl}}
}`;
