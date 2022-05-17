# ArkoWrapper

[![Python 3.9](https://img.shields.io/badge/Python-3.8_|_3.9_|_3.10-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/release/python-390/)
[![GitHub file size in bytes](https://img.shields.io/github/languages/code-size/ArkoClub/ArkoWrapper?label=Size&logo=hack-the-box&logoColor=white&style=flat-square)](https://github.com/ArkoClub/ArkoWrapper)
[![PyPI](https://img.shields.io/pypi/v/arko-wrapper?color=%233775A9&label=PyPI&logo=pypi&logoColor=white&style=flat-square)](https://pypi.org/project/arko-wrapper/)
[![Codacy Badge](https://img.shields.io/codacy/grade/78563bf9a5304851a73684c34a30e2b3?label=Code%20Quality&logo=Codacy&style=flat-square)](https://www.codacy.com/gh/ArkoClub/ArkoWrapper/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ArkoClub/ArkoWrapper&amp;utm_campaign=Badge_Grade)
[![License](https://img.shields.io/github/license/ArkoClub/ArkoWrapper?label=License&style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB0PSIxNjUxMjEyODQ2ODY0IiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjE1MDAiIHdpZHRoPSIzMDAiIGhlaWdodD0iMzAwIj48cGF0aCBkPSJNOTQ3LjIgOTIxLjZsLTg3MC40IDBjLTQyLjM0MjQgMC03Ni44LTM0LjQ1NzYtNzYuOC03Ni44bDAtNjY1LjZjMC00Mi4zNDI0IDM0LjQ1NzYtNzYuOCA3Ni44LTc2LjhsODcwLjQgMGM0Mi4zNDI0IDAgNzYuOCAzNC40NTc2IDc2LjggNzYuOGwwIDY2NS42YzAgNDIuMzQyNC0zNC40NTc2IDc2LjgtNzYuOCA3Ni44ek03Ni44IDE1My42Yy0xNC4xMzEyIDAtMjUuNiAxMS40Njg4LTI1LjYgMjUuNmwwIDY2NS42YzAgMTQuMTMxMiAxMS40Njg4IDI1LjYgMjUuNiAyNS42bDg3MC40IDBjMTQuMTMxMiAwIDI1LjYtMTEuNDY4OCAyNS42LTI1LjZsMC02NjUuNmMwLTE0LjEzMTItMTEuNDY4OC0yNS42LTI1LjYtMjUuNmwtODcwLjQgMHoiIHAtaWQ9IjE1MDEiIGZpbGw9IiNmZmZmZmYiPjwvcGF0aD48cGF0aCBkPSJNNDg2LjQgMzA3LjJsLTMwNy4yIDBjLTE0LjEzMTIgMC0yNS42LTExLjQ2ODgtMjUuNi0yNS42czExLjQ2ODgtMjUuNiAyNS42LTI1LjZsMzA3LjIgMGMxNC4xMzEyIDAgMjUuNiAxMS40Njg4IDI1LjYgMjUuNnMtMTEuNDY4OCAyNS42LTI1LjYgMjUuNnoiIHAtaWQ9IjE1MDIiIGZpbGw9IiNmZmZmZmYiPjwvcGF0aD48cGF0aCBkPSJNNDg2LjQgNDYwLjhsLTMwNy4yIDBjLTE0LjEzMTIgMC0yNS42LTExLjQ2ODgtMjUuNi0yNS42czExLjQ2ODgtMjUuNiAyNS42LTI1LjZsMzA3LjIgMGMxNC4xMzEyIDAgMjUuNiAxMS40Njg4IDI1LjYgMjUuNnMtMTEuNDY4OCAyNS42LTI1LjYgMjUuNnoiIHAtaWQ9IjE1MDMiIGZpbGw9IiNmZmZmZmYiPjwvcGF0aD48cGF0aCBkPSJNNDg2LjQgNTYzLjJsLTMwNy4yIDBjLTE0LjEzMTIgMC0yNS42LTExLjQ2ODgtMjUuNi0yNS42czExLjQ2ODgtMjUuNiAyNS42LTI1LjZsMzA3LjIgMGMxNC4xMzEyIDAgMjUuNiAxMS40Njg4IDI1LjYgMjUuNnMtMTEuNDY4OCAyNS42LTI1LjYgMjUuNnoiIHAtaWQ9IjE1MDQiIGZpbGw9IiNmZmZmZmYiPjwvcGF0aD48cGF0aCBkPSJNNDg2LjQgNjY1LjZsLTMwNy4yIDBjLTE0LjEzMTIgMC0yNS42LTExLjQ2ODgtMjUuNi0yNS42czExLjQ2ODgtMjUuNiAyNS42LTI1LjZsMzA3LjIgMGMxNC4xMzEyIDAgMjUuNiAxMS40Njg4IDI1LjYgMjUuNnMtMTEuNDY4OCAyNS42LTI1LjYgMjUuNnoiIHAtaWQ9IjE1MDUiIGZpbGw9IiNmZmZmZmYiPjwvcGF0aD48cGF0aCBkPSJNNDM1LjIgNzY4bC0yNTYgMGMtMTQuMTMxMiAwLTI1LjYtMTEuNDY4OC0yNS42LTI1LjZzMTEuNDY4OC0yNS42IDI1LjYtMjUuNmwyNTYgMGMxNC4xMzEyIDAgMjUuNiAxMS40Njg4IDI1LjYgMjUuNnMtMTEuNDY4OCAyNS42LTI1LjYgMjUuNnoiIHAtaWQ9IjE1MDYiIGZpbGw9IiNmZmZmZmYiPjwvcGF0aD48cGF0aCBkPSJNOTE4LjY4MTYgMzM1LjA1MjhsLTQxLjYyNTYtMzAuMjU5Mi0xNS45MjMyLTQ4Ljk0NzItNTEuNDU2IDAtNDEuNjI1Ni0zMC4yNTkyLTQxLjYyNTYgMzAuMjU5Mi01MS40NTYgMC0xNS45MjMyIDQ4Ljk0NzItNDEuNjI1NiAzMC4yNTkyIDE1LjkyMzIgNDguOTQ3Mi0xNS45MjMyIDQ4Ljk0NzIgNDEuNjI1NiAzMC4yNTkyIDYuNzU4NCAyMC43ODcyYy0wLjEwMjQgMC44MTkyLTAuMTAyNCAxLjU4NzItMC4xMDI0IDIuNDA2NGwwIDI1NmMwIDEwLjM0MjQgNi4yNDY0IDE5LjcxMiAxNS44MjA4IDIzLjY1NDRzMjAuNTgyNCAxLjc5MiAyNy45MDQtNS41Mjk2bDU4LjY3NTItNTguNjc1MiA1OC42NzUyIDU4LjY3NTJjNC45MTUyIDQuOTE1MiAxMS40MTc2IDcuNTI2NCAxOC4xMjQ4IDcuNDc1MiAzLjI3NjggMCA2LjYwNDgtMC42MTQ0IDkuNzc5Mi0xLjk0NTYgOS41NzQ0LTMuOTQyNCAxNS44MjA4LTEzLjMxMiAxNS44MjA4LTIzLjY1NDRsMC0yNTZjMC0wLjgxOTItMC4wNTEyLTEuNjM4NC0wLjEwMjQtMi40MDY0bDYuNzU4NC0yMC43ODcyIDQxLjYyNTYtMzAuMjU5Mi0xNS45MjMyLTQ4Ljk0NzIgMTUuOTIzMi00OC45NDcyek02NzcuNTI5NiAzNTQuNjExMmwyNC45ODU2LTE4LjE3NiA5LjU3NDQtMjkuMzg4OCAzMC45MjQ4IDAgMjQuOTg1Ni0xOC4xNzYgMjQuOTg1NiAxOC4xNzYgMzAuOTI0OCAwIDkuNTc0NCAyOS4zODg4IDI0Ljk4NTYgMTguMTc2LTkuNTc0NCAyOS4zODg4IDkuNTc0NCAyOS4zODg4LTI0Ljk4NTYgMTguMTc2LTkuNTc0NCAyOS4zODg4LTMwLjkyNDggMC0yNC45ODU2IDE4LjE3Ni0yNC45ODU2LTE4LjE3Ni0zMC45MjQ4IDAtOS41NzQ0LTI5LjM4ODgtMjQuOTg1Ni0xOC4xNzYgOS41NzQ0LTI5LjM4ODgtOS41NzQ0LTI5LjM4ODh6TTc4Ni4xMjQ4IDY0Ny40NzUyYy05Ljk4NC05Ljk4NC0yNi4yMTQ0LTkuOTg0LTM2LjE5ODQgMGwtMzMuMDc1MiAzMy4wNzUyIDAtMTY4LjQ0OCA5LjU3NDQgMCA0MS42MjU2IDMwLjI1OTIgNDEuNjI1Ni0zMC4yNTkyIDkuNTc0NCAwIDAgMTY4LjQ0OC0zMy4wNzUyLTMzLjA3NTJ6IiBwLWlkPSIxNTA3IiBmaWxsPSIjZmZmZmZmIj48L3BhdGg+PC9zdmc+)](./LICENSE)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/ArkoClub/ArkoWrapper/PyPI%20publish%20CI?label=Build&logo=github-actions&logoColor=white&style=flat-square)](https://github.com/ArkoClub/ArkoWrapper/actions/workflows/pypi-publish.yml)

[![Wakatime](https://wakatime.com/badge/user/570bddef-37a7-4738-b1f7-969ab95c4cc9/project/b409f839-97d4-4b01-9eb5-66e44ee122a7.svg?style=flat-square&label=WakaTime)](https://wakatime.com/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/arko-wrapper?color=91A4ED&label=Downloads&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB0PSIxNjUyMjYwMDAwMjU3IiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjUxNjIiIHdpZHRoPSI1MDAiIGhlaWdodD0iNTAwIj48cGF0aCBkPSJNOTU3LjU0MzkzNyA5NjEuMTMxMTM3IDYyLjg2NTI4MSA5NjEuMTMxMTM3IDYyLjg2NTI4MSA2NTUuNTA5NDg1IDE4OC4yOTgwNjIgNjU1LjUwOTQ4NSAxODguMjk4MDYyIDg1OS4yNDI1ODYgODMyLjE2MDI3NSA4NTkuMjQyNTg2IDgzMi4xNjAyNzUgNjU1LjUwOTQ4NSA5NTcuNTQzOTM3IDY1NS41MDk0ODVaIiBwLWlkPSI1MTYzIiBmaWxsPSIjZmZmZmZmIj48L3BhdGg%2BPHBhdGggZD0iTTc1My4yNzg3MTcgMzYzLjI3ODgxNyA1MTAuMTgwMDUgNzkwLjc0MTQ0NSAyNjcuMDMzMjg3IDM2My4yNzg4MTdaIiBwLWlkPSI1MTY0IiBmaWxsPSIjZmZmZmZmIj48L3BhdGg%2BPHBhdGggZD0iTTQzNC44OTEzMiA2NC4zNTY3NWwxNTAuNTI4MzQyIDAgMCAzMDAuMjU5NTI4LTE1MC41MjgzNDIgMCAwLTMwMC4yNTk1MjhaIiBwLWlkPSI1MTY1IiBmaWxsPSIjZmZmZmZmIj48L3BhdGg%2BPC9zdmc%2B&style=flat-square)](https://pypi.org/project/arko-wrapper/)
[![View](https://img.shields.io/badge/dynamic/json?color=7AA3CC&label=View&query=%24.value&url=https%3A%2F%2Fapi.countapi.xyz%2Fhit%2FArkoClub%2FArkoWrapper&style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB0PSIxNjUyMjU5MTQ0MjAzIiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjM0NjkiIHdpZHRoPSIzMDAiIGhlaWdodD0iMzAwIj48cGF0aCBkPSJNNTEyIDQxNmE5NiA5NiAwIDEgMCAwIDE5MiA5NiA5NiAwIDAgMCAwLTE5MnogbTUxMS45NTIgMTAyLjA2NGMtMC4wMTYtMC40NDgtMC4wNjQtMC44NjQtMC4wOTYtMS4yOTZhOC4xNiA4LjE2IDAgMCAwLTAuMDgtMC42NTZjMC0wLjMyLTAuMDY0LTAuNjI0LTAuMTI4LTAuOTI4LTAuMDMyLTAuMzY4LTAuMDY0LTAuNzM2LTAuMTI4LTEuMDg4LTAuMDMyLTAuMDQ4LTAuMDMyLTAuMDk2LTAuMDMyLTAuMTQ0YTM5LjQ4OCAzOS40ODggMCAwIDAtMTAuNzA0LTIxLjUzNmMtMzIuNjcyLTM5LjYxNi03MS41MzYtNzQuODgtMTExLjA0LTEwNy4wNzItODUuMDg4LTY5LjM5Mi0xODIuNDMyLTEyNy40MjQtMjg5Ljg1Ni0xNTAuOC02Mi4xMTItMTMuNTA0LTEyNC41NzYtMTQuMDY0LTE4Ny4wMDgtMi42NC01Ni43ODQgMTAuMzg0LTExMS41MDQgMzItMTYyLjcyIDU4Ljc4NC04MC4xNzYgNDEuOTItMTUzLjM5MiA5OS42OTYtMjE3LjE4NCAxNjQuNDgtMTEuODA4IDExLjk4NC0yMy41NTIgMjQuMjI0LTM0LjI4OCAzNy4yNDgtMTQuMjg4IDE3LjMyOC0xNC4yODggMzcuODcyIDAgNTUuMjE2IDMyLjY3MiAzOS42MTYgNzEuNTIgNzQuODQ4IDExMS4wNCAxMDcuMDU2IDg1LjEyIDY5LjM5MiAxODIuNDQ4IDEyNy40MDggMjg5Ljg4OCAxNTAuNzg0IDYyLjA5NiAxMy41MDQgMTI0LjYwOCAxNC4wOTYgMTg3LjAwOCAyLjY1NiA1Ni43NjgtMTAuNCAxMTEuNDg4LTMyIDE2Mi43MzYtNTguNzY4IDgwLjE3Ni00MS45MzYgMTUzLjM3Ni05OS42OTYgMjE3LjE4NC0xNjQuNDggMTEuNzkyLTEyIDIzLjUzNi0yNC4yMjQgMzQuMjg4LTM3LjI0OCA1LjcxMi01Ljg3MiA5LjQ1Ni0xMy40NCAxMC43MDQtMjEuNTY4bDAuMDMyLTAuMTI4YTEyLjU5MiAxMi41OTIgMCAwIDAgMC4xMjgtMS4wODhjMC4wNjQtMC4zMDQgMC4wOTYtMC42MjQgMC4xMjgtMC45MjhsMC4wOC0wLjY1NiAwLjA5Ni0xLjI4YzAuMDMyLTAuNjU2IDAuMDQ4LTEuMjk2IDAuMDQ4LTEuOTUybC0wLjA5Ni0xLjk2OHpNNTEyIDcwNGMtMTA2LjAzMiAwLTE5Mi04NS45NTItMTkyLTE5MnM4NS45NTItMTkyIDE5Mi0xOTIgMTkyIDg1Ljk2OCAxOTIgMTkyYzAgMTA2LjA0OC04NS45NjggMTkyLTE5MiAxOTJ6IiBwLWlkPSIzNDcwIiBmaWxsPSIjZmZmZmZmIj48L3BhdGg+PC9zdmc+)](https://github.com/ArkoClub/ArkoWrapper)

给你的Python迭代器加上魔法

### Building...

[//]: # ([![Google Code Style]&#40;https://img.shields.io/badge/Code%20Style-Google-9cf?style=flat-square&logoColor=white&logo=data:image/svg+xml;base64,PHN2ZyB0PSIxNjUxMjE1NDU2MjUwIiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjMzMTEiIHdpZHRoPSIzMDAiIGhlaWdodD0iMzAwIj48cGF0aCBkPSJNOTU4LjE3IDQ0Ny40TDc2MC42OSAyNDkuOTJsLTY1LjgyIDY1LjgzIDE5Ny40NyAxOTcuNDdMNjk0Ljg3IDcxMC43bDY1LjgyIDY1LjgyIDE5Ny40OC0xOTcuNDcgNjUuODMtNjUuODN6TTI2My4zIDI0OS45Mkw2NS44MiA0NDcuNCAwIDUxMy4yMmw2NS44MiA2NS44M0wyNjMuMyA3NzYuNTJsNjUuODItNjUuODItMTk3LjQ3LTE5Ny40OCAxOTcuNDctMTk3LjQ3ek0zNDMuMjQ3IDk0OS40ODNMNTkwLjk2IDUyLjE5bDg5LjcyIDI0Ljc2OC0yNDcuNzEzIDg5Ny4yOTV6IiBwLWlkPSIzMzEyIiBmaWxsPSIjZmZmZmZmIj48L3BhdGg+PC9zdmc+&#41;]&#40;https://google.github.io/styleguide/pyguide.html&#41;)
