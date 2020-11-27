// Get from backend api later
const authors = ["Stephan King"," Meryl Streep", "Lady Gaga"]

const authorList = document.getElementById("authors")

html = ""
authors.forEach(a=>{
    html+= `<input type='checkbox' name='${a}' value='${a}'>`
    html+=`  <label for="${a}">${a}</label><br>`
})
authorList.innerHTML=html

// Get genre from backend
const genreList = document.getElementById("genre")

const genre = ["murder","adventure","mystery"]
html = ""
genre.forEach(a=>{
    html+= `<input type='checkbox' name='${a}' value='${a}'>`
    html+=`  <label for="${a}">${a}</label><br>`
})
genreList.innerHTML = html