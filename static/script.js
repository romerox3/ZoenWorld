const mundo = document.getElementById('mundo');
const mundoWidth = 500;
const mundoHeight = 500;

function actualizarMundo() {
    fetch('/estado')
        .then(response => response.json())
        .then(estado => {
            console.log('Estado recibido:', estado);
            console.log('Entidades:', estado.entidades);
            console.log('Comida:', estado.recursos.comida);
            console.log('Arboles:', estado.recursos.arboles);
            mundo.innerHTML = '';

            // Dibujar comida
            estado.recursos.comida.forEach(comida => {
                const comidaElement = document.createElement('div');
                comidaElement.className = 'comida';
                const x = (comida[0] + 250) * (mundoWidth / 500);
                const y = (250 - comida[1]) * (mundoHeight / 500);
                comidaElement.style.left = `${Math.max(0, Math.min(mundoWidth - 5, x))}px`;
                comidaElement.style.top = `${Math.max(0, Math.min(mundoHeight - 5, y))}px`;
                mundo.appendChild(comidaElement);
            });

            // Dibujar árboles
            estado.recursos.arboles.forEach(arbol => {
                const arbolElement = document.createElement('div');
                arbolElement.className = 'arbol';
                const x = (arbol[0] + 250) * (mundoWidth / 500);
                const y = (250 - arbol[1]) * (mundoHeight / 500);
                arbolElement.style.left = `${Math.max(0, Math.min(mundoWidth - 10, x))}px`;
                arbolElement.style.top = `${Math.max(0, Math.min(mundoHeight - 10, y))}px`;
                mundo.appendChild(arbolElement);
            });

            // Dibujar entidades
            if (estado.entidades && estado.entidades.length > 0) {
                console.log('Número de entidades:', estado.entidades.length);
                estado.entidades.forEach(entidad => {
                    console.log('Dibujando entidad:', entidad);
                    const entidadElement = document.createElement('div');
                    entidadElement.className = 'entidad';

                    const x = (entidad.posicion_x + 250) * (mundoWidth / 500);
                    const y = (250 - entidad.posicion_y) * (mundoHeight / 500);

                    entidadElement.style.left = `${Math.max(0, Math.min(mundoWidth - 10, x))}px`;
                    entidadElement.style.top = `${Math.max(0, Math.min(mundoHeight - 10, y))}px`;

                    const energyPercentage = entidad.energia / 100;
                    const red = Math.floor(255 * (1 - energyPercentage));
                    const green = Math.floor(255 * energyPercentage);
                    entidadElement.style.backgroundColor = `rgb(${red}, ${green}, 0)`;

                    entidadElement.title = `${entidad.nombre}
                    Energía: ${entidad.energia.toFixed(2)}
                    Puntuación: ${entidad.puntuacion.toFixed(2)}
                    Recompensa promedio: ${entidad.recompensa_promedio.toFixed(4)}
                    Pérdida promedio: ${entidad.perdida_promedio.toFixed(4)}
                    Epsilon: ${entidad.epsilon.toFixed(4)}`;
                    mundo.appendChild(entidadElement);
                });
            } else {
                console.log('No se encontraron entidades en el estado');
            }
            actualizarInfoEntrenamiento(estado.entidades);
        })
        .catch(error => console.error('Error al obtener el estado:', error));
}

function actualizarInfoEntrenamiento(entidades) {
    const infoEntrenamiento = document.getElementById('info-entrenamiento');
    infoEntrenamiento.innerHTML = '<h2>Información de Entrenamiento</h2>';
    
    const descripciones = {
        puntuacion: "La puntuación total acumulada por la entidad. Un valor más alto indica un mejor desempeño general.",
        recompensa_promedio: "El promedio de las últimas 100 recompensas. Un valor positivo y creciente indica que la entidad está aprendiendo a tomar mejores decisiones.",
        perdida_promedio: "El promedio de las últimas 100 pérdidas de entrenamiento. Un valor decreciente indica que la red neuronal está mejorando sus predicciones.",
        epsilon: "La probabilidad de que la entidad tome una acción aleatoria. Disminuye con el tiempo, lo que significa que la entidad confía más en su aprendizaje."
    };
    
    entidades.forEach(entidad => {
        const entidadInfo = document.createElement('div');
        entidadInfo.innerHTML = `
            <h3>${entidad.nombre}</h3>
            <p><strong>Puntuación:</strong> ${entidad.puntuacion.toFixed(2)}</p>
            <p class="descripcion">${descripciones.puntuacion}</p>
            <p><strong>Recompensa promedio:</strong> ${entidad.recompensa_promedio.toFixed(4)}</p>
            <p class="descripcion">${descripciones.recompensa_promedio}</p>
            <p><strong>Pérdida promedio:</strong> ${entidad.perdida_promedio.toFixed(4)}</p>
            <p class="descripcion">${descripciones.perdida_promedio}</p>
            <p><strong>Epsilon:</strong> ${entidad.epsilon.toFixed(4)}</p>
            <p class="descripcion">${descripciones.epsilon}</p>
        `;
        infoEntrenamiento.appendChild(entidadInfo);
    });
}

setInterval(actualizarMundo, 1000);
actualizarMundo();