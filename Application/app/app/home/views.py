from flask import render_template
from . import home


@home.route("/")
def homepage():
    """
    Render the homepage template on the / route
    """
    author_tokens = {'Clive Barker' : '[CLIVE BARKER]', 'J. K. Rowling' : '[J.K. ROWLING]', 'Stephen King' : '[STEPHEN KING]', 'Théophile Gautier' : '[THEOPHILE GAUTIER]',
               'James H. Hyslop' : '[JAMES H HYSLOP]', 'Lord Edward Bulwer-Lytton' : '[LORD EDWARD BULWER-LYTTON]', 'A. T. Quiller-Couch' : '[A. T. QUILLER-COUCH]',
               'Mrs. Margaret Oliphant' : '[MRS. MARGARET OLIPHANT]', 'Ernest Theodor Amadeus Hoffmann' : '[ERNEST THEODOR AMADEUS HOFFMAN]', 'Erckmann-Chatrian' : '[ERCKMANN-CHATRAIN]',
               'Fiona Macleod' : '[FIONA MACLEOD]', 'Amelia B. Edwards' : '[AMELIA B. EDWARDS]', 'H. B. Marryatt' : '[H. B. MARRYATT]', 'Thomas Hardy' : '[THOMAS HARDY]',
               'Montague Rhodes James' : '[MONTAGUE RHODES JAMES]', 'Fitz-James O\'Brien' : '[FITZ-JAMES O\'BRIEN', 'James Stephen' : '[JAMES STEPHEN]', 'Alfred Lord Tennyson' : '[ALFRED LORD TENNYSON]',
               'Amelia Edwards' : '[AMELIA EDWARDS]', 'Edward Bulwer-Lytton' : '[EDWARD BULWER-LYTTON]', 'Erckmann Chatrian' : '[ERCKMANN CHATRIAN]', 'Latifa al-Zayya' : '[LATIFA AL-ZAYYA]',
               'M. R. James' : '[M. R. JAMES]', 'Paul Brandis' : '[PAUL BRANDIS]', 'Brain Evenson' : '[BRAIN EVENSON]', 'Elliott O\'Donnell' : '[ELLIOTT O\'DONNELL]',
               'Joseph, Sheridan Le Fanu' : '[JOSEPH, SHERIDAN LE FANU]', 'Edgar Allan Poe' : '[EDGAR ALLEN POE]', 'Bram Stoker' : '[BRAM STOKER]', 'Algernon Blackwood' :'[ALGERNON BLACKWOOD]',
               'Miles Klee' : '[MILES KLEE]', 'Nnedi Okorador' : '[NNEDI OKORADOR]', 'Sofia Samatar' : '[SOFIA SAMATAR]', 'Franz Kafka' : '[FRANZ KAFKA]', 'Laird Barron' : '[LAIRD BARRON]',
               'Nathan Ballingrud' : '[NATHAN BALLINGRUD]', 'Nellie Bly' : '[NELLIE BLY]', 'William Hop Hodgson' : '[WILLIAM HOP HODGSON]', 'Ambrose Bierce' : '[AMBROSE BIERCE]',
               'Kelly Link' : '[KELLY LINK]', 'Arthur Machen' : '[ARTHUR MACHEN]', 'George Sylvester Viereck' : '[GEORGE SYLVESTER VIERECK]', 'Robert Chambers' : '[ROBERT CHAMBERS]',
               'John Meade Falkner' : '[JOHN MEADE FALKNER]', 'Ann Radcliffe' : '[ANN RADCLIFFE]', 'Howard Lovecraft' : '[HOWARD LOVECRAFT]', 'Louis Stevenson' : '[LOUIS STEVENSON]',
               'Edith Birkhead' : '[EDITH BIRKHEAD]', 'Jeff Vandermeer' : '[JEFF VANDERMEER]', 'Henry James' : '[HENRY JAMES]', 'John William Polidori' : '[JOHN WILLIAM POLIDORI]',
               'Bob Holland' : '[BOB HOLLAND]', 'Oliver Onions' : '[OLIVER ONIONS]'}
    author_inputs = f''''''
    for author in author_tokens:
        author_inputs += f'''
        <div class="checkbox-container author">
            <span class="input-title">{author}</span>
            <label class="checkbox-label">
                <input type="checkbox" value="{author}">
                <span class="checkbox-custom rectangular"></span>
            </label>
        </div>
        '''

    subgenre_tokens = ['Vampire', 'Ghost', 'Horror', 'Comedic Horror', 'Murder',
                   'Werewolf', 'Apocalypse','Haunted House', 'Witch', 'Hell',
                   'Alien', 'Gore', 'Monster']
    genre_inputs = f''''''
    for genre in subgenre_tokens:
        genre_inputs += f'''
        <div class="checkbox-container genre">
            <span class="input-title">{genre}</span>
            <label class="checkbox-label">
                <input type="checkbox" value="{genre}">
                <span class="checkbox-custom rectangular"></span>
            </label>
        </div>
        '''
    return render_template("page/home/index.html", title="Welcome", author_selection=author_inputs, genre_selection=genre_inputs)


@home.route("/dashboard")
def dashboard():
    """
    Render the dashboard template on the /dashboard route
    """
    return render_template("page/home/dashboard.html", title="Dashboard")
